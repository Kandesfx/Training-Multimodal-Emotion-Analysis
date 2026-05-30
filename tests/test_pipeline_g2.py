import sys
from pathlib import Path
import os
from typing import Tuple, Dict, Any, List
import cv2
import numpy as np
import scipy.io.wavfile as wav

# Thêm project root vào PYTHONPATH
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Sử dụng thư mục tạm thời riêng để test
os.environ["EDS_DATA_DIR"] = str(Path(project_root) / "data_test_g2")

from fastapi.testclient import TestClient
from backend.main import app
from backend.database.local_db import Base, engine, SessionLocal
from backend.database.models import Video, Clip
from backend.config import settings

client = TestClient(app)

def create_mock_video_and_audio() -> Tuple[str, str]:
    """Sinh ra một file video MP4 test và audio WAV test siêu nhẹ để chạy thử pipeline."""
    test_dir = Path(settings.DATA_DIR) / "mock_assets"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = test_dir / "test_video.mp4"
    audio_path = test_dir / "test_audio.wav"
    
    # 1. Tạo video 5 giây, 25 FPS, độ phân giải 640x480 (chứa hình tròn vẽ mặt cười giả lập)
    width, height = 640, 480
    fps = 25
    duration = 5.0
    total_frames = int(fps * duration)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for i in range(total_frames):
        # Vẽ một khung hình nền đen
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Vẽ khuôn mặt cười đơn giản:
        # Mặt hình tròn màu vàng
        cv2.circle(frame, (320, 240), 100, (0, 255, 255), -1)
        # Mắt trái
        cv2.circle(frame, (280, 200), 10, (0, 0, 0), -1)
        # Mắt phải
        cv2.circle(frame, (360, 200), 10, (0, 0, 0), -1)
        # Miệng cười
        cv2.ellipse(frame, (320, 260), (40, 20), 0, 0, 180, (0, 0, 0), 5)
        
        out.write(frame)
    out.release()
    
    # 2. Tạo audio WAV 5 giây, sample rate 16000Hz (sin wave đơn giản)
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sin wave 440Hz
    data = np.sin(2 * np.pi * 440 * t) * 32767
    data = data.astype(np.int16)
    wav.write(str(audio_path), sr, data)
    
    return str(video_path.resolve()), str(audio_path.resolve())

def test_pipeline_g2():
    print("=== BẮT ĐẦU KIỂM THỬ PIPELINE AI GIAI ĐOẠN 2 (E2E) ===")
    
    # Dọn dẹp database cũ
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Tạo asset mock local để test offline nhanh chóng và ổn định
    video_file, audio_file = create_mock_video_and_audio()
    print(f"Đã sinh mock video: {video_file}")
    print(f"Đã sinh mock audio: {audio_file}")
    
    # Thêm bản ghi video local vào DB để chạy thử
    db = SessionLocal()
    db_video = Video(
        title="Video Test Pipeline",
        movie_name="Bo Phim Test",
        file_path=video_file,
        status="pending"
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    video_id = db_video.id
    db.close()
    
    # Chạy thử API kích hoạt xử lý thực tế
    print(f"\n1. Gọi API khởi chạy xử lý AI cho video ID {video_id}...")
    response = client.post(f"/api/videos/{video_id}/process")
    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    
    # Đợi 5 giây để background task thực hiện toàn bộ pipeline AI ngầm
    print("Đang đợi background task xử lý toàn bộ 7 AI Services (Downloader -> Splitter -> Extractor -> Transcriber -> Analyzer -> Scorer)...")
    import time
    time.sleep(6.0)
    
    # 2. Xác minh trạng thái Video sau xử lý
    print("\n2. Kiểm tra thông tin video sau xử lý...")
    response = client.get(f"/api/videos/{video_id}")
    assert response.status_code == 200
    video_data = response.json()
    print(f"Trạng thái cuối cùng: {video_data['status']}")
    print(f"Tổng số clips phân cảnh tạo ra: {video_data['total_clips']}")
    print(f"Approved clips: {video_data['approved_clips']}")
    
    # Vì video mock của chúng ta dài 5 giây, PySceneDetect phát hiện 1 cảnh, 
    # và thời lượng 5 giây nằm trong khoảng 3s-15s nên sẽ được cắt thành đúng 1 clip.
    assert video_data["status"] == "completed"
    assert video_data["total_clips"] == 1
    
    # 3. Xác minh các clips và metadata AI đi kèm
    print("\n3. Lấy thông tin clips để xác minh metadata AI...")
    response = client.get(f"/api/clips/?video_id={video_id}")
    assert response.status_code == 200
    clips = response.json()
    assert len(clips) == 1
    
    clip = clips[0]
    print(f"Thông tin clip chi tiết:")
    print(f"  - Clip Path: {clip['clip_path']}")
    print(f"  - Số frame xử lý: {clip['num_frames']}")
    print(f"  - Số mặt detect được: {clip['num_faces']}")
    print(f"  - Transcript tiếng Việt: '{clip['transcript']}'")
    print(f"  - Loa chính: {clip['speaker_id']}")
    print(f"  - Cảm xúc chiến thắng (Ensemble): {clip['predicted_emotion']}")
    print(f"  - Độ tin cậy (Confidence): {clip['confidence']}")
    print(f"  - Đồng thuận mô hình (Agreement): {clip['agreement']}")
    print(f"  - Điểm chất lượng (Quality Score): {clip['quality_score']}")
    print(f"  - Trạng thái định tuyến duyệt: {clip['status']}")
    
    # Kiểm tra các assert logic
    assert clip["num_frames"] > 0
    assert clip["predicted_emotion"] in ["happy", "sad", "angry", "surprise", "fear", "disgust", "neutral"]
    assert 0.0 <= clip["confidence"] <= 1.0
    assert "/" in clip["agreement"]
    assert 0.0 <= clip["quality_score"] <= 1.0
    assert clip["status"] in ["approved", "pending", "rejected"]
    
    # 4. Kiểm tra Dashboard Stats
    print("\n4. Xác minh Dashboard Stats mới...")
    response = client.get("/api/labels/stats")
    assert response.status_code == 200
    stats = response.json()
    print(f"Stats: Total Clips = {stats['total_clips']}, Approved = {stats['approved_clips']}")
    assert stats["total_clips"] == 1
    
    print("\n=== HOÀN TẤT KIỂM THỬ GIAI ĐOẠN 2: THÀNH CÔNG MỸ MÃN! ===")

if __name__ == "__main__":
    try:
        test_pipeline_g2()
    finally:
        # Dọn dẹp connection pool và thư mục test
        from backend.database.local_db import engine
        engine.dispose()
        
        import shutil
        test_dir = Path(project_root) / "data_test_g2"
        if test_dir.exists():
            try:
                shutil.rmtree(test_dir)
                print("Đã dọn dẹp thư mục mock assets và test database.")
            except Exception as e:
                print(f"Cảnh báo dọn dẹp: {e}")
