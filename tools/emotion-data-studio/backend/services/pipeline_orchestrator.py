import os
import json
import traceback
import cv2
from sqlalchemy.orm import Session
from backend.database.models import Video, Clip
from backend.config import settings

# Import tất cả các services của AI Pipeline
from backend.services.downloader import VideoDownloader
from backend.services.scene_splitter import SceneSplitter
from backend.services.smart_segmenter import SmartSegmenter
from backend.services.face_extractor import FaceExtractor
from backend.services.audio_extractor import AudioExtractor
from backend.services.transcriber import SpeechTranscriber
from backend.services.emotion_analyzer import EmotionAnalyzer
from backend.services.quality_scorer import QualityScorer

class PipelineOrchestrator:
    """Bộ điều phối trung tâm quản lý chạy tuần tự toàn bộ AI Pipeline từ đầu đến cuối."""
    
    # Singleton pattern — tránh tạo lại các service objects mỗi lần chạy
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.downloader = VideoDownloader()
        self.splitter = SceneSplitter()
        self.smart_segmenter = SmartSegmenter(
            face_scan_fps=settings.SMART_FACE_SCAN_FPS,
            face_confidence=settings.SMART_FACE_CONFIDENCE,
            max_missing_face_gap=settings.SMART_MAX_MISSING_FACE_GAP,
            min_duration=settings.MIN_CLIP_DURATION,
            max_duration=settings.MAX_CLIP_DURATION,
            target_duration=settings.SMART_TARGET_CLIP_DURATION,
            silence_threshold_db=settings.SMART_SILENCE_THRESHOLD_DB,
            silence_min_duration=settings.SMART_SILENCE_MIN_DURATION,
            max_dialogue_extension=settings.SMART_MAX_DIALOGUE_EXTENSION,
            vad_mode=settings.SMART_VAD_MODE,
        )
        self.face_extractor = FaceExtractor()
        self.audio_extractor = AudioExtractor()
        self.transcriber = SpeechTranscriber()
        self.analyzer = EmotionAnalyzer()
        self.quality_scorer = QualityScorer()
        from backend.utils.resource_manager import resource_manager
        self.resource_plan = resource_manager.apply()
        print(f"⚙️ [Resource] device={self.resource_plan.device}, CPU threads={self.resource_plan.cpu_threads}, FFmpeg threads={self.resource_plan.ffmpeg_threads}, GPU={self.resource_plan.gpu_name or 'none'}")
        self._initialized = True

    def run_pipeline(self, video_id: str, db: Session, progress_callback=None):
        """Khởi chạy toàn bộ Pipeline xử lý video gốc và lưu trữ các clips phân cảnh."""
        print(f"🎬 [Pipeline] Bắt đầu chạy Orchestrator cho Video ID: {video_id}")
        
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            print(f"❌ [Pipeline] Không tìm thấy video ID {video_id} trong DB")
            return
            
        try:
            # --- STAGE 1: DOWNLOAD VIDEO (nếu là YouTube URL) ---
            video.status = "processing"
            db.commit()
            
            video_file_path = video.file_path
            
            if video.source_url and not video_file_path:
                print(f"📥 [Pipeline] Stage 1: Đang tải video từ YouTube URL: {video.source_url}...")
                
                last_pct = -1
                def download_hook(d):
                    nonlocal last_pct
                    if d['status'] == 'downloading':
                        total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                        downloaded = d.get('downloaded_bytes', 0)
                        if total > 0:
                            pct = int(downloaded / total * 100)
                            if pct != last_pct:
                                last_pct = pct
                                if progress_callback:
                                    progress_callback("download", downloaded, total, f"📥 [Download] Đang tải: {pct}% ({downloaded / 1024 / 1024:.1f}MB / {total / 1024 / 1024:.1f}MB)")
                        else:
                            megabytes_downloaded = int(downloaded / 1024 / 1024)
                            if megabytes_downloaded % 5 == 0 and megabytes_downloaded != last_pct:
                                last_pct = megabytes_downloaded
                                if progress_callback:
                                    progress_callback("download", downloaded, 0, f"📥 [Download] Đang tải: {downloaded / 1024 / 1024:.1f}MB (chưa rõ dung lượng)")
                    elif d['status'] == 'finished':
                        if progress_callback:
                            progress_callback("download", 100, 100, "📥 [Download] Đã tải xong video, đang hoàn thiện tệp...")

                if progress_callback:
                    progress_callback("download", 0, 100, f"📥 [Download] Bắt đầu tải video từ YouTube URL...")

                download_res = self.downloader.download(video.source_url, progress_hook=download_hook)
                
                # Cập nhật thông tin video sau tải
                video.file_path = download_res["file_path"]
                video.duration_sec = download_res["duration_sec"]
                video.resolution = download_res["resolution"]
                if download_res.get("title") and (not video.title or video.title == "Unknown" or video.title == "Untitled"):
                    video.title = download_res["title"]
                video_file_path = download_res["file_path"]
                db.commit()
                print(f"✅ [Pipeline] Stage 1: Tải thành công video. Độ dài: {video.duration_sec}s")
                if progress_callback:
                    progress_callback("title_retrieved", 0, 0, video.title)
                    progress_callback("download", 100, 100, f"📥 [Download] Tải thành công video. Độ dài: {video.duration_sec}s")
            else:
                if progress_callback:
                    # Đảm bảo UI cập nhật tên file cục bộ nếu nó là Unknown/Untitled
                    if video.title and progress_callback:
                        progress_callback("title_retrieved", 0, 0, video.title)
                    progress_callback("download", 100, 100, "📥 [Download] Sử dụng file video cục bộ có sẵn")
                
            if not video_file_path or not os.path.exists(video_file_path):
                raise FileNotFoundError(f"Không tìm thấy file video cục bộ tại: {video_file_path}")
                
            # --- STAGE 2: DETECT & SPLIT SCENES ---
            print("✂️ [Pipeline] Stage 2: Đang tự động phát hiện chuyển cảnh và cắt video...")
            if progress_callback:
                progress_callback("scene_split", 0, 100, "✂️ [Scene Split] Bắt đầu phát hiện chuyển cảnh...")
            scenes = self.splitter.detect_scenes(video_file_path, threshold=settings.SCENE_THRESHOLD)
            
            # Logic Fallback: Nếu không phát hiện điểm chuyển cảnh nào, coi cả video là 1 scene duy nhất
            if len(scenes) == 0:
                print("⚠️ [Pipeline] Không phát hiện điểm chuyển cảnh. Tự động tạo 1 scene duy nhất cho toàn bộ video.")
                duration = video.duration_sec
                if not duration:
                    # Lấy duration bằng OpenCV
                    cap = cv2.VideoCapture(video_file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps > 0 else 5.0
                    cap.release()
                
                scenes = [{
                    "scene_index": 0,
                    "start_time": 0.0,
                    "end_time": duration,
                    "duration": duration
                }]
                
            print(f"ℹ️ [Pipeline] Tổng số {len(scenes)} cảnh thô. Bắt đầu smart segmentation theo mặt người/hội thoại...")
            if progress_callback:
                progress_callback("scene_split", 35, 100, f"✂️ [Smart Segment] Phân tích mặt người và vùng hội thoại trong {len(scenes)} cảnh...")

            try:
                smart_scenes = self.smart_segmenter.build_segments(video_file_path, scenes, video_id=video_id)
                if smart_scenes:
                    scenes_for_split = smart_scenes
                    print(f"✅ [Pipeline] SmartSegmenter tạo {len(scenes_for_split)} candidate clips face/dialogue-aware")
                    if progress_callback:
                        progress_callback("scene_split", 70, 100, f"✂️ [Smart Segment] Tạo {len(scenes_for_split)} đoạn có mặt/hội thoại")
                else:
                    scenes_for_split = scenes
                    print("⚠️ [Pipeline] SmartSegmenter không tạo candidate nào, fallback scene-only")
                    if progress_callback:
                        progress_callback("scene_split", 70, 100, "✂️ [Smart Segment] Không thấy đoạn có mặt rõ, fallback cắt theo scene")
            except Exception as smart_err:
                scenes_for_split = scenes
                print(f"⚠️ [Pipeline] SmartSegmenter lỗi, fallback scene-only: {smart_err}")
                if progress_callback:
                    progress_callback("scene_split", 70, 100, f"✂️ [Smart Segment] Fallback scene-only: {smart_err}")
            
            # Cắt video theo smart candidates và lọc theo thời lượng
            clips_metadata = self.splitter.split_video(
                video_file_path,
                scenes_for_split,
                video_id,
                min_duration=settings.MIN_CLIP_DURATION,
                max_duration=settings.MAX_CLIP_DURATION,
            )
            total_clips = len(clips_metadata)
            
            video.total_clips = total_clips
            db.commit()
            print(f"✅ [Pipeline] Stage 2: Hoàn tất. Đã tạo ra {total_clips} clips hợp lệ để xử lý AI")
            if progress_callback:
                progress_callback("scene_split", 100, 100, f"✂️ [Scene Split] Hoàn tất. Đã tạo ra {total_clips} clips hợp lệ")
            
            # --- STAGES 3 & 4: MULTI-TRACK PROCESSING & MULTI-MODEL LABELING ---
            
            # Pre-warm: Load TẤT CẢ AI models một lần trước khi bắt đầu xử lý clips
            # Tránh việc load rời rạc gây chậm cho clip đầu tiên
            if total_clips > 0:
                print("🔥 [Pipeline] Pre-warming AI models trước khi xử lý clips...")
                if progress_callback:
                    progress_callback("prewarm", 0, 100, "🔥 [Pre-warm] Đang nạp AI models lên bộ nhớ...")
                from backend.ai_models.model_manager import model_manager
                loaded, failed = model_manager.prewarm_models()
                if failed:
                    for status in model_manager.status():
                        if status.error:
                            print(f"⚠️ [Pre-warm] Model {status.key} lỗi: {status.error[:300]}")
                if progress_callback:
                    progress_callback("prewarm", 100, 100, 
                                      f"🔥 [Pre-warm] Hoàn tất: {loaded} models loaded, {failed} failed")
            
            import time as _time
            
            for idx, clip_meta in enumerate(clips_metadata):
                clip_index = clip_meta["clip_index"]
                clip_path = clip_meta["clip_path"]
                clip_id = f"{video_id}_clip_{clip_index}"
                
                print(f"\n⚡ [Pipeline] Xử lý Clip {clip_index + 1}/{total_clips}: {clip_id}")
                
                if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                    raise Exception("Pipeline bị hủy bởi người dùng")
                
                try:
                    # 1. Trích xuất hình ảnh (Face Extraction & Tracking)
                    print(f"  [Visual] Detect khuôn mặt...")
                    if progress_callback:
                        progress_callback("face_detect", idx, total_clips, f"👤 [Visual] Detect khuôn mặt clip {idx + 1}/{total_clips}...")
                    face_res = self.face_extractor.extract_faces_from_clip(clip_path, clip_id)
                    if progress_callback:
                        progress_callback("face_detect", idx + 1, total_clips, "")
                    
                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")

                    # 2. Trích xuất âm thanh (Audio extraction & MFCC)
                    print(f"  [Audio] Tách WAV & tính MFCC...")
                    if progress_callback:
                        progress_callback("audio_extract", idx, total_clips, f"🔊 [Audio] Tách WAV & tính MFCC clip {idx + 1}/{total_clips}...")
                    audio_res = self.audio_extractor.extract_audio_from_clip(clip_path, clip_id)
                    if progress_callback:
                        progress_callback("audio_extract", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 3. Chuyển đổi giọng nói thành văn bản (Speech-to-Text)
                    print(f"  [Text] Chạy Whisper Speech-to-Text tiếng Việt...")
                    if progress_callback:
                        progress_callback("transcribe", idx, total_clips, f"📝 [Text] Chạy Whisper Speech-to-Text clip {idx + 1}/{total_clips}...")
                    text_res = self.transcriber.transcribe_audio_clip(audio_res["audio_path"], clip_id)
                    if text_res.get("warning"):
                        print(f"  ⚠️ [Text] {text_res.get('warning')}")
                    elif not text_res.get("transcript"):
                        print("  ⚠️ [Text] Không nhận diện được lời thoại trong clip này")
                    if progress_callback:
                        progress_callback("transcribe", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 4. Phân tích đa phương thức kết hợp (Multi-model Ensemble Voting)
                    print(f"  [AI Ensemble] Chạy Ensemble Voting cảm xúc...")
                    if progress_callback:
                        progress_callback("emotion_label", idx, total_clips, f"🎭 [AI Ensemble] Chạy Ensemble Voting cảm xúc clip {idx + 1}/{total_clips}...")
                    ai_res = self.analyzer.analyze_clip(
                        face_images=face_res["cropped_face_paths"],
                        transcript=text_res["transcript"],
                        audio_path=audio_res["audio_path"]
                    )
                    if progress_callback:
                        progress_callback("emotion_label", idx + 1, total_clips, "")

                    if progress_callback and not progress_callback("check_cancel", 0, 0, ""):
                        raise Exception("Pipeline bị hủy bởi người dùng")
                    
                    # 5. Chấm điểm chất lượng và Định tuyến trạng thái duyệt tự động
                    print(f"  [Quality] Chấm điểm chất lượng...")
                    if progress_callback:
                        progress_callback("quality_score", idx, total_clips, f"⭐ [Quality] Chấm điểm chất lượng clip {idx + 1}/{total_clips}...")
                    quality_res = self.quality_scorer.calculate_score(
                        confidence=ai_res["confidence"],
                        agreement_str=ai_res["agreement"],
                        sampled_frames_count=face_res["num_frames"],
                        cropped_faces_count=face_res["main_track_len"],
                        audio_clarity=audio_res["audio_clarity"]
                    )
                    if progress_callback:
                        progress_callback("quality_score", idx + 1, total_clips, "")
                    
                    # 6. Lưu thông tin clip chi tiết vào cơ sở dữ liệu
                    db_clip = Clip(
                        id=clip_id,
                        video_id=video_id,
                        clip_index=clip_index,
                        start_time=clip_meta["start_time"],
                        end_time=clip_meta["end_time"],
                        duration=clip_meta["duration"],
                        clip_path=clip_path,
                        num_frames=face_res["num_frames"],
                        num_faces=face_res["num_faces"],
                        transcript=text_res["transcript"],
                        speaker_id=text_res["main_speaker"],
                        quality_score=quality_res["quality_score"],
                        status=quality_res["status"],
                        predicted_emotion=ai_res["predicted_emotion"],
                        confidence=ai_res["confidence"],
                        agreement=ai_res["agreement"],
                        has_incongruity=ai_res["has_incongruity"],
                        all_scores=ai_res["all_scores"],
                        per_model_scores={
                            **(ai_res.get("per_model_scores") or {}),
                            "segment": clip_meta.get("segment_metadata", {}),
                            "face_extraction": {
                                "detector": face_res.get("detector"),
                                "face_paths": face_res.get("cropped_face_paths", []),
                                "detections_path": face_res.get("detections_path"),
                                "track_count": face_res.get("track_count", 0),
                            },
                            "audio_features": {
                                "audio_path": audio_res.get("audio_path"),
                                "audio_clarity": audio_res.get("audio_clarity"),
                                "has_speech_energy": audio_res.get("has_speech_energy"),
                            },
                        }
                    )
                    db.add(db_clip)
                    db.commit()
                    print(f"✅ [Pipeline] Xử lý thành công Clip #{clip_index}. Trạng thái định tuyến: {quality_res['status'].upper()}")
                    
                except Exception as clip_err:
                    print(f"❌ [Pipeline] Lỗi khi xử lý Clip #{clip_index}: {clip_err}")
                    traceback.print_exc()
                    if "bị hủy bởi người dùng" in str(clip_err):
                        raise clip_err
                    # Vẫn tiếp tục xử lý các clips khác nếu 1 clip gặp lỗi
                    continue
            
            # Cập nhật video hoàn tất
            video.status = "completed"
            # Cập nhật số lượng approved_clips tự động
            approved_count = db.query(Clip).filter(
                Clip.video_id == video_id,
                Clip.status.in_(["approved", "auto_approved"])
            ).count()
            video.approved_clips = approved_count
            db.commit()
            print(f"\n🎉 [Pipeline] HOÀN TẤT THÀNH CÔNG toàn bộ quy trình cho Video ID: {video_id}")
            
            # Đảm bảo tất cả các stage kết thúc đều hiển thị 100%
            if progress_callback:
                for stage_key in ["download", "scene_split", "face_detect", "audio_extract", "transcribe", "emotion_label", "quality_score"]:
                    progress_callback(stage_key, 100, 100, "")

        except Exception as e:
            print(f"❌ [Pipeline] Gặp lỗi nghiêm trọng dừng pipeline của Video {video_id}: {e}")
            traceback.print_exc()
            if "bị hủy bởi người dùng" in str(e) or "cancelled" in str(e).lower():
                video.status = "cancelled"
            else:
                video.status = "error"
            db.commit()
            raise e
