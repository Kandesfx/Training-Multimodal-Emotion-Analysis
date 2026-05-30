# 🏗️ KIẾN TRÚC HỆ THỐNG — MULTIMODAL EMOTION ANALYSIS
## Tài liệu Tham chiếu Duy nhất cho AI Agent Coding

> **MỤC ĐÍCH:** Tài liệu này là nguồn chân lý duy nhất (Single Source of Truth) cho toàn bộ dự án. Mọi AI Agent coding phải đọc file này trước khi thực hiện bất kỳ thay đổi nào.

> **NGUYÊN TẮC:** Mọi thông số, interface, contract đều được ghi rõ ràng. KHÔNG có chỗ nào cần "đoán" hoặc "tự quyết định".

---

## MỤC LỤC

1. [Tổng quan Hệ thống](#1-tổng-quan-hệ-thống)
2. [Cấu trúc Thư mục](#2-cấu-trúc-thư-mục)
3. [Data Pipeline (tools/)](#3-data-pipeline)
4. [Module Video — CNN (models/video_module.py)](#4-module-video)
5. [Module Audio — CNN 1D + BiLSTM (models/audio_module.py)](#5-module-audio)
6. [Module Text — PhoBERT + BiLSTM (models/text_module.py)](#6-module-text)
7. [Fusion Hub (models/fusion.py)](#7-fusion-hub)
8. [Multimodal Classifier (models/classifier.py)](#8-multimodal-classifier)
9. [Dataset & DataLoader (training/dataset.py)](#9-dataset--dataloader)
10. [Training Pipeline (training/)](#10-training-pipeline)
11. [Backend API (backend/)](#11-backend-api)
12. [Frontend UI (frontend/)](#12-frontend-ui)
13. [Data Schema & Formats](#13-data-schema--formats)
14. [Chiến lược 2 Giai đoạn Training](#14-chiến-lược-2-giai-đoạn)
15. [Testing & Verification](#15-testing--verification)
16. [Dependency Map](#16-dependency-map)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Mô tả

Hệ thống phân tích cảm xúc đa phương thức (Multimodal Emotion Analysis) từ video. Nhận đầu vào là file video, phân tích đồng thời 3 kênh thông tin (hình ảnh khuôn mặt, giọng nói, lời thoại), kết hợp kết quả qua Fusion Hub và đưa ra nhận định cảm xúc.

### 1.2 Sơ đồ Kiến trúc Tổng thể

```
                        ┌──────────────┐
                        │  INPUT:      │
                        │  Video File  │
                        │  (.mp4/.avi) │
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │   PREPROCESSING     │
                    │   (backend/services)│
                    └──┬──────┬───────┬───┘
                       │      │       │
            ┌──────────▼┐ ┌───▼────┐ ┌▼──────────┐
            │  Frames   │ │ Audio  │ │ Text      │
            │  (faces)  │ │ (.wav) │ │ (Whisper) │
            └─────┬─────┘ └───┬────┘ └─────┬─────┘
                  │            │            │
            ┌─────▼─────┐ ┌───▼─────┐ ┌────▼──────┐
            │  VIDEO     │ │  AUDIO  │ │  TEXT     │
            │  MODULE    │ │  MODULE │ │  MODULE   │
            │            │ │         │ │           │
            │ ResNet50   │ │CNN1D    │ │PhoBERT   │
            │ +Temporal  │ │+BiLSTM  │ │+BiLSTM   │
            │  LSTM      │ │         │ │           │
            └─────┬──────┘ └───┬─────┘ └────┬──────┘
                  │            │             │
            [B, 512]     [B, 512]      [B, 512]
                  │            │             │
                  └──────┬─────┴─────────────┘
                         │
                  ┌──────▼──────┐
                  │  FUSION HUB │
                  │             │
                  │ GĐ1: Concat │
                  │ + MLP       │
                  │             │
                  │ GĐ2: Cross- │
                  │ Attention   │
                  │ + Gating    │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │ CLASSIFIER  │
                  │ FC→512→256  │
                  │ →7 classes  │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  OUTPUT:    │
                  │  7 emotions │
                  │  + scores   │
                  └─────────────┘
```

### 1.3 Nhãn Cảm xúc (7 classes)

| Index | English   | Tiếng Việt   | Emoji |
|-------|-----------|--------------|-------|
| 0     | happy     | Vui vẻ       | 😊    |
| 1     | sad       | Buồn bã      | 😢    |
| 2     | angry     | Tức giận     | 😠    |
| 3     | fear      | Sợ hãi       | 😨    |
| 4     | surprise  | Ngạc nhiên   | 😲    |
| 5     | disgust   | Ghê tởm      | 🤢    |
| 6     | neutral   | Trung tính   | 😐    |

### 1.4 Tech Stack

| Thành phần      | Công nghệ                          | Phiên bản tối thiểu |
|------------------|-------------------------------------|----------------------|
| Language         | Python                              | 3.10+                |
| DL Framework     | PyTorch                             | 2.1+                 |
| NLP              | transformers (PhoBERT)              | 4.36+                |
| Vision           | torchvision                         | 0.16+                |
| Audio            | librosa, torchaudio                 | 0.10+                |
| Face Detection   | facenet-pytorch (MTCNN)             | 2.5+                 |
| STT              | openai-whisper                      | 20231117+            |
| Backend          | Flask + Flask-CORS                  | 3.0+                 |
| Frontend         | HTML5 + Vanilla JS + CSS3           | —                    |
| Charts           | Chart.js (CDN)                      | 4.4+                 |
| GPU Runtime      | Google Colab Pro (A100)             | —                    |

---

## 2. CẤU TRÚC THƯ MỤC

```
BCDA/                              # Project root
│
├── docs/                          # 📚 Tài liệu
│   ├── architecture/
│   │   └── ARCHITECTURE.md        # ★ File này — tài liệu kiến trúc chính
│   ├── research/
│   │   ├── Analysis.md            # Phân tích kỹ thuật ban đầu
│   │   ├── phan_tich_training_da_phuong_thuc.md
│   │   └── yeu_cau_de_tai.md      # Yêu cầu đề tài gốc
│   └── reports/
│       └── (báo cáo đồ án sẽ viết ở đây)
│
├── data/                          # 📊 Dữ liệu
│   ├── raw/                       # Video gốc tải về
│   │   └── {movie_name}/
│   │       └── {episode}.mp4
│   ├── processed/                 # Đã xử lý
│   │   ├── video_frames/          # Frames khuôn mặt đã crop
│   │   │   └── {clip_id}/
│   │   │       ├── frame_0000.jpg
│   │   │       ├── frame_0001.jpg
│   │   │       └── ...
│   │   ├── audio/                 # Audio đã tách
│   │   │   └── {clip_id}.wav
│   │   └── transcripts/           # Text từ Whisper
│   │       └── {clip_id}.json
│   ├── labels/                    # Nhãn cảm xúc
│   │   ├── annotations.csv        # File nhãn chính
│   │   └── pre_labels.csv         # Nhãn gợi ý từ DeepFace
│   └── splits/                    # Chia tập train/val/test
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── tools/                         # 🔧 Pipeline khai thác dữ liệu
│   ├── __init__.py
│   ├── download_videos.py         # Tải video (yt-dlp)
│   ├── scene_splitter.py          # Cắt scene (PySceneDetect)
│   ├── extract_components.py      # Tách audio + text (FFmpeg + Whisper)
│   ├── face_detector.py           # Detect & crop khuôn mặt (MTCNN)
│   ├── pre_labeler.py             # Pre-label (DeepFace)
│   └── build_dataset.py           # Orchestrator: chạy toàn bộ pipeline
│
├── models/                        # 🧠 Kiến trúc mô hình
│   ├── __init__.py
│   ├── video_module.py            # CNN (ResNet50) + Temporal LSTM
│   ├── audio_module.py            # CNN 1D + BiLSTM
│   ├── text_module.py             # PhoBERT + BiLSTM
│   ├── fusion.py                  # Fusion Hub (Concat / Cross-Attention)
│   └── classifier.py              # Wrapper tổng: MultimodalClassifier
│
├── training/                      # 🏋️ Pipeline training
│   ├── __init__.py
│   ├── config.py                  # ★ Config trung tâm (đã tạo)
│   ├── dataset.py                 # Custom PyTorch Dataset
│   ├── dataloader.py              # DataLoader factory + augmentation
│   ├── trainer.py                 # Training loop + logging
│   ├── evaluate.py                # Metrics + confusion matrix
│   ├── losses.py                  # Loss functions (CE, Incongruity, Focal)
│   └── utils.py                   # Seed, checkpoint, early stopping
│
├── backend/                       # 🖥️ Flask API server
│   ├── __init__.py
│   ├── app.py                     # Flask app factory
│   ├── config.py                  # Server config
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py              # Tất cả API routes
│   │   └── schemas.py             # Request/Response schemas
│   └── services/
│       ├── __init__.py
│       ├── preprocessor.py        # Video → frames/audio/text
│       ├── predictor.py           # Load model + inference
│       └── file_handler.py        # Upload/download management
│
├── frontend/                      # 🎨 Web UI
│   ├── index.html                 # Main page
│   ├── css/
│   │   └── styles.css             # Dark theme + glassmorphism
│   ├── js/
│   │   ├── app.js                 # Main controller
│   │   ├── upload.js              # Video upload handler
│   │   ├── api.js                 # Backend API calls
│   │   ├── results.js             # Results rendering
│   │   └── charts.js              # Chart.js visualizations
│   └── assets/
│       └── (icons, images)
│
├── notebooks/                     # 📓 Jupyter notebooks (Colab)
│   ├── 01_data_pipeline.ipynb
│   ├── 02_train_video.ipynb
│   ├── 03_train_audio.ipynb
│   ├── 04_train_text.ipynb
│   ├── 05_train_fusion.ipynb
│   └── 06_evaluation.ipynb
│
├── tests/                         # 🧪 Unit tests
│   ├── __init__.py
│   ├── test_video_module.py
│   ├── test_audio_module.py
│   ├── test_text_module.py
│   ├── test_fusion.py
│   ├── test_classifier.py
│   ├── test_dataset.py
│   └── test_api.py
│
├── checkpoints/                   # 💾 Model weights
│   └── (auto-saved during training)
│
├── logs/                          # 📈 Training logs
│   └── (tensorboard logs)
│
├── requirements.txt               # Dependencies (đã tạo)
├── README.md                      # Project overview
└── .gitignore
```

---

## 3. DATA PIPELINE

### 3.1 Tổng quan Pipeline

```
YouTube URL → yt-dlp → raw .mp4
                          │
                    PySceneDetect
                          │
                    clips (5-15s each)
                          │
              ┌───────────┼───────────┐
              │           │           │
         FFmpeg       FFmpeg      Whisper
         (frames)    (audio)     (STT vi)
              │           │           │
          MTCNN       .wav 16kHz   .json text
         (crop face)      │           │
              │           │           │
         DeepFace     ────┴───────────┘
         (pre-label)
              │
         annotations.csv → Label Studio → final labels
```

### 3.2 File: `tools/download_videos.py`

```
PURPOSE: Tải video từ YouTube bằng yt-dlp
INPUT:   File CSV/JSON chứa danh sách URL, hoặc URL đơn lẻ
OUTPUT:  File .mp4 trong data/raw/{movie_name}/

CLASS: VideoDownloader
  __init__(self, output_dir: str = "data/raw")
  
  METHODS:
    download_single(url: str, movie_name: str) -> Path
      - Dùng yt-dlp Python API (không subprocess)
      - Format: mp4, chất lượng 720p (format: "bestvideo[height<=720]+bestaudio/best[height<=720]")
      - Output: data/raw/{movie_name}/{video_id}.mp4
      - Return đường dẫn file đã tải
    
    download_from_list(csv_path: str) -> List[Path]
      - Đọc CSV có columns: url, movie_name
      - Gọi download_single cho từng URL
      - Skip nếu file đã tồn tại
      - Log progress với tqdm
    
    get_metadata(url: str) -> dict
      - Trả về: title, duration, channel, upload_date
      - Không tải video, chỉ lấy info

CLI USAGE:
  python tools/download_videos.py --url "https://youtube.com/..." --name "ve_nha_di_con"
  python tools/download_videos.py --csv "data/video_list.csv"
```

### 3.3 File: `tools/scene_splitter.py`

```
PURPOSE: Tự động cắt video thành các clip ngắn theo scene
INPUT:   File .mp4
OUTPUT:  Nhiều file .mp4 clips trong data/processed/clips/{movie_name}/

CLASS: SceneSplitter
  __init__(self, 
           min_duration: float = 3.0,    # Bỏ clip < 3 giây
           max_duration: float = 15.0,   # Bỏ clip > 15 giây
           threshold: float = 27.0)      # ContentDetector threshold
  
  METHODS:
    split_video(video_path: Path, output_dir: Path) -> List[dict]
      - Dùng scenedetect.ContentDetector
      - Tách scenes, lọc theo min/max duration
      - Lưu clips bằng FFmpeg
      - Return list of {clip_id, start_time, end_time, duration, path}
    
    split_all(input_dir: Path, output_dir: Path) -> pd.DataFrame
      - Xử lý tất cả video trong input_dir
      - Return DataFrame tổng hợp all clips
      - Save manifest CSV: data/processed/clips_manifest.csv

CLI USAGE:
  python tools/scene_splitter.py --input "data/raw/ve_nha_di_con/" --output "data/processed/clips/"
```

### 3.4 File: `tools/extract_components.py`

```
PURPOSE: Từ mỗi clip, tách ra 3 thành phần: audio, transcript text
INPUT:   File .mp4 clip
OUTPUT:  - data/processed/audio/{clip_id}.wav
         - data/processed/transcripts/{clip_id}.json

CLASS: ComponentExtractor
  __init__(self,
           sample_rate: int = 16000,
           whisper_model: str = "medium",   # "medium" cho accuracy tốt trên A100
           whisper_language: str = "vi")
  
  METHODS:
    extract_audio(clip_path: Path, output_path: Path) -> Path
      - FFmpeg: mp4 → wav (16kHz, mono, PCM 16-bit)
      - Normalize volume
      - Return path to wav
    
    transcribe(audio_path: Path) -> dict
      - Whisper inference
      - Return: {
          "text": "full transcription",
          "segments": [
            {"start": 0.0, "end": 2.5, "text": "Anh ơi..."},
            ...
          ],
          "language": "vi",
          "confidence": 0.92
        }
    
    process_clip(clip_path: Path) -> dict
      - Gọi extract_audio + transcribe
      - Save transcript JSON
      - Return metadata dict
    
    process_all(clips_dir: Path) -> pd.DataFrame
      - Xử lý tất cả clips
      - Parallel processing nếu có thể (nhưng cẩn thận GPU memory với Whisper)
      - Return DataFrame

CLI USAGE:
  python tools/extract_components.py --clips-dir "data/processed/clips/" --whisper-model "medium"
```

### 3.5 File: `tools/face_detector.py`

```
PURPOSE: Detect khuôn mặt trong video frames, crop và lưu
INPUT:   File .mp4 clip
OUTPUT:  data/processed/video_frames/{clip_id}/frame_XXXX.jpg

CLASS: FaceFrameExtractor
  __init__(self,
           frame_interval: float = 0.5,   # Lấy 1 frame mỗi 0.5 giây
           image_size: int = 224,          # Resize face crop
           min_face_size: int = 60,        # Bỏ khuôn mặt quá nhỏ
           face_confidence: float = 0.95,  # MTCNN confidence threshold
           device: str = "cuda")
  
  ATTRIBUTES:
    self.detector: MTCNN (from facenet_pytorch)
      - image_size=224, margin=20, keep_all=False
      - select_largest=True (lấy khuôn mặt lớn nhất nếu nhiều người)
  
  METHODS:
    extract_frames(video_path: Path) -> List[np.ndarray]
      - OpenCV VideoCapture
      - Lấy frame theo interval (mỗi 0.5s)
      - Return list of BGR frames
    
    detect_and_crop(frame: np.ndarray) -> Optional[np.ndarray]
      - MTCNN detect
      - Crop face region + margin
      - Resize → 224x224
      - Return RGB face image hoặc None nếu không detect được
    
    process_clip(clip_path: Path, output_dir: Path) -> dict
      - extract_frames → detect_and_crop cho mỗi frame
      - Lưu frames: {output_dir}/{clip_id}/frame_0000.jpg, frame_0001.jpg, ...
      - Return: {clip_id, num_frames, num_faces_detected, face_detection_rate}
    
    process_all(clips_dir: Path, output_dir: Path) -> pd.DataFrame
      - Batch processing tất cả clips
      - Log clips có face_detection_rate < 0.5 (cảnh quay xa, không có người)
      - Return DataFrame

QUAN TRỌNG:
  - Mỗi clip phải có ít nhất 4 frames có khuôn mặt để sử dụng
  - Clips không đủ faces sẽ bị đánh dấu "insufficient_faces" trong manifest

CLI USAGE:
  python tools/face_detector.py --clips-dir "data/processed/clips/" --output "data/processed/video_frames/"
```

### 3.6 File: `tools/pre_labeler.py`

```
PURPOSE: Dùng DeepFace để tự động gợi ý nhãn cảm xúc cho mỗi clip
INPUT:   Frames đã crop trong data/processed/video_frames/
OUTPUT:  data/labels/pre_labels.csv

CLASS: PreLabeler
  __init__(self, 
           detector_backend: str = "skip",    # Đã crop face rồi, skip detection
           emotion_model: str = "default")
  
  METHODS:
    analyze_frame(frame_path: Path) -> dict
      - DeepFace.analyze(actions=["emotion"])
      - Return: {"dominant_emotion": "happy", "scores": {"happy": 85, "sad": 3, ...}}
    
    analyze_clip(clip_id: str, frames_dir: Path) -> dict
      - Phân tích tất cả frames trong clip
      - Aggregate bằng weighted average (frame giữa clip có weight cao hơn)
      - Return: {
          "clip_id": "clip_001",
          "predicted_emotion": "happy",
          "confidence": 0.82,
          "emotion_scores": {"happy": 0.82, "sad": 0.05, ...},
          "needs_review": False,        # True nếu confidence < 0.6
          "agreement_rate": 0.85        # Tỷ lệ frames đồng ý với emotion chính
        }
    
    process_all(frames_base_dir: Path) -> pd.DataFrame
      - Xử lý tất cả clips
      - Save CSV: data/labels/pre_labels.csv
      - Log statistics: số clip/emotion, confidence distribution
      - Return DataFrame

OUTPUT CSV SCHEMA (pre_labels.csv):
  clip_id | predicted_emotion | confidence | needs_review | agreement_rate | happy | sad | angry | fear | surprise | disgust | neutral

CLI USAGE:
  python tools/pre_labeler.py --frames-dir "data/processed/video_frames/"
```

### 3.7 File: `tools/build_dataset.py`

```
PURPOSE: Orchestrator — chạy toàn bộ pipeline từ đầu đến cuối
INPUT:   CSV danh sách URL hoặc thư mục video
OUTPUT:  Dataset hoàn chỉnh sẵn sàng training

CLASS: DatasetBuilder
  __init__(self, config: ProjectConfig)
  
  METHODS:
    run_full_pipeline(source: str, source_type: str = "csv") -> Path
      - Step 1: download_videos (nếu source_type == "csv" hoặc "url")
      - Step 2: split scenes
      - Step 3: extract audio + transcribe
      - Step 4: detect & crop faces
      - Step 5: pre-label emotions
      - Step 6: generate train/val/test splits (70/15/15)
      - Return path to splits directory
    
    create_splits(labels_path: Path, ratios: tuple = (0.7, 0.15, 0.15)) -> dict
      - Stratified split (giữ tỷ lệ emotions cân bằng)
      - Save: data/splits/train.csv, val.csv, test.csv
      - Return: {"train": N, "val": N, "test": N}
    
    generate_statistics() -> dict
      - Tổng số clips, phân bố emotion, duration stats
      - Export chart ảnh vào docs/reports/

CLI USAGE:
  python tools/build_dataset.py --source "data/video_list.csv" --source-type csv
  python tools/build_dataset.py --source "data/raw/" --source-type directory
```

---

## 4. MODULE VIDEO

### File: `models/video_module.py`

```
PURPOSE: Trích xuất đặc trưng biểu cảm khuôn mặt từ chuỗi video frames
BACKBONE: ResNet50 (pretrained ImageNet) → Temporal LSTM
INPUT:  Tensor shape (batch_size, num_frames, 3, 224, 224)
OUTPUT: Tensor shape (batch_size, 512)

CLASS: VideoModule(nn.Module)
  __init__(self,
           backbone: str = "resnet50",
           pretrained: bool = True,
           feature_dim: int = 512,
           num_frames: int = 16,
           temporal_hidden: int = 256,
           temporal_layers: int = 1,
           dropout: float = 0.3,
           freeze_backbone_layers: int = 6)  # Freeze first 6 layers of ResNet
  
  ARCHITECTURE:
    self.backbone = torchvision.models.resnet50(pretrained=True)
      - Xóa layer FC cuối (self.backbone.fc = nn.Identity())
      - Output: 2048-dim per frame
      - Freeze layers [conv1, bn1, layer1, layer2, layer3, first 2 blocks of layer4]
        để tiết kiệm memory và tránh overfit trên dataset nhỏ
    
    self.spatial_projection = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3)
    )
    
    self.temporal_lstm = nn.LSTM(
        input_size=512,
        hidden_size=256,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
        dropout=0.0  # Chỉ 1 layer nên không dùng LSTM dropout
    )
    # BiLSTM output: 256*2 = 512 → match feature_dim
    
    self.temporal_attention = nn.Sequential(
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )
    # Attention pooling: học frame nào quan trọng nhất
    
    self.output_norm = nn.LayerNorm(512)
  
  forward(self, x: Tensor) -> Tensor:
    # x: (B, T, 3, 224, 224)
    B, T = x.shape[:2]
    
    # 1. Flatten batch & time → extract features per frame
    x = x.view(B * T, 3, 224, 224)
    x = self.backbone(x)           # (B*T, 2048)
    x = self.spatial_projection(x) # (B*T, 512)
    
    # 2. Reshape back → temporal sequence
    x = x.view(B, T, 512)         # (B, T, 512)
    
    # 3. Temporal LSTM
    x, _ = self.temporal_lstm(x)   # (B, T, 512)  [bidirectional: 256*2]
    
    # 4. Attention pooling over time
    attn_weights = self.temporal_attention(x)  # (B, T, 1)
    attn_weights = F.softmax(attn_weights, dim=1)
    x = (x * attn_weights).sum(dim=1)         # (B, 512)
    
    # 5. Output normalization
    x = self.output_norm(x)                    # (B, 512)
    return x

TRANSFORM (cho inference và training):
  train_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.RandomCrop(224),
      transforms.RandomHorizontalFlip(0.5),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
  ])
  
  val_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
  ])
```

---

## 5. MODULE AUDIO

### File: `models/audio_module.py`

```
PURPOSE: Trích xuất đặc trưng cảm xúc từ giọng nói
INPUT:  Tensor shape (batch_size, n_features, time_steps) — MFCC features
OUTPUT: Tensor shape (batch_size, 512)

FEATURE EXTRACTION (trong dataset.py, không trong module):
  - librosa.feature.mfcc(y, sr=16000, n_mfcc=40)          → (40, T)
  - librosa.feature.delta(mfcc)                             → (40, T)
  - librosa.feature.delta(mfcc, order=2)                    → (40, T)
  - Stack → (120, T)
  - Pad/truncate T → max_time_steps = 300 (≈ 15 giây ở hop_length=512)
  - Normalize theo z-score (per feature)

CLASS: AudioModule(nn.Module)
  __init__(self,
           n_features: int = 120,      # 40 MFCC + 40 delta + 40 delta-delta
           feature_dim: int = 512,
           cnn_channels: list = [64, 128, 256],
           lstm_hidden: int = 256,
           lstm_layers: int = 2,
           dropout: float = 0.3)
  
  ARCHITECTURE:
    self.cnn_blocks = nn.ModuleList()
    # 3 CNN 1D blocks, mỗi block: Conv1d → BatchNorm → ReLU → MaxPool → Dropout
    
    Block 1: Conv1d(120, 64,  kernel_size=5, padding=2) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    Block 2: Conv1d(64,  128, kernel_size=5, padding=2) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    Block 3: Conv1d(128, 256, kernel_size=3, padding=1) → BN → ReLU → MaxPool(2) → Dropout(0.2)
    # Sau 3 lần MaxPool(2): time_steps giảm 8x (300 → 37)
    
    self.lstm = nn.LSTM(
        input_size=256,
        hidden_size=256,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.3
    )
    # Output: 256*2 = 512
    
    self.attention = nn.Sequential(
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )
    
    self.output_projection = nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.3)
    )
  
  forward(self, x: Tensor) -> Tensor:
    # x: (B, 120, T)  — MFCC features
    
    # 1. CNN feature extraction
    for block in self.cnn_blocks:
        x = block(x)                 # (B, 256, T//8)
    
    # 2. Transpose for LSTM: (B, T//8, 256)
    x = x.permute(0, 2, 1)
    
    # 3. BiLSTM temporal modeling
    x, _ = self.lstm(x)              # (B, T//8, 512)
    
    # 4. Attention pooling
    attn = F.softmax(self.attention(x), dim=1)  # (B, T//8, 1)
    x = (x * attn).sum(dim=1)                   # (B, 512)
    
    # 5. Output projection
    x = self.output_projection(x)                # (B, 512)
    return x
```

---

## 6. MODULE TEXT

### File: `models/text_module.py`

```
PURPOSE: Trích xuất đặc trưng ngữ nghĩa cảm xúc từ lời thoại tiếng Việt
INPUT:  Dict từ PhoBERT tokenizer: {input_ids: (B, seq_len), attention_mask: (B, seq_len)}
OUTPUT: Tensor shape (batch_size, 512)

TOKENIZATION (trong dataset.py):
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
  encoded = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

CLASS: TextModule(nn.Module)
  __init__(self,
           phobert_model: str = "vinai/phobert-base-v2",
           feature_dim: int = 512,
           lstm_hidden: int = 256,
           lstm_layers: int = 2,
           dropout: float = 0.3,
           freeze_phobert: bool = True,    # GĐ1: freeze toàn bộ PhoBERT
           unfreeze_last_n: int = 0)       # GĐ2: có thể unfreeze 2-4 layers cuối
  
  ARCHITECTURE:
    self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
      - Output: (B, seq_len, 768)
      - Freeze tất cả params (GĐ1) hoặc unfreeze last N layers (GĐ2)
    
    self.projection = nn.Sequential(
        nn.Linear(768, 512),
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    
    self.lstm = nn.LSTM(
        input_size=512,
        hidden_size=256,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        dropout=0.3
    )
    # Output: 256*2 = 512
    
    self.attention = nn.Sequential(
        nn.Linear(512, 128),
        nn.Tanh(),
        nn.Linear(128, 1)
    )
    
    self.output_norm = nn.LayerNorm(512)
  
  forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    # 1. PhoBERT encoding
    with torch.no_grad() if self.freeze_phobert else contextmanager():
        phobert_out = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x = phobert_out.last_hidden_state  # (B, seq_len, 768)
    
    # 2. Project 768 → 512
    x = self.projection(x)                  # (B, seq_len, 512)
    
    # 3. BiLSTM
    x, _ = self.lstm(x)                     # (B, seq_len, 512)
    
    # 4. Masked attention pooling (ignore padding)
    attn = self.attention(x)                # (B, seq_len, 1)
    attn = attn.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
    attn = F.softmax(attn, dim=1)
    x = (x * attn).sum(dim=1)              # (B, 512)
    
    # 5. Output norm
    x = self.output_norm(x)                 # (B, 512)
    return x

NOTE:
  - PhoBERT frozen = không update weights → tiết kiệm VRAM, train nhanh
  - Tuy frozen nhưng BiLSTM phía trên vẫn học được representation tốt
  - GĐ2: unfreeze 2-4 layers cuối để fine-tune sâu hơn
```

---

## 7. FUSION HUB

### File: `models/fusion.py`

```
PURPOSE: Kết hợp features từ 3 modules thành unified representation
INPUT:  3 tensors, mỗi tensor shape (B, 512)
OUTPUT: Tensor shape (B, 512)

═══════════════════════════════════════
GĐ1: ConcatFusion (đơn giản, chắc chắn)
═══════════════════════════════════════

CLASS: ConcatFusion(nn.Module)
  __init__(self,
           input_dim: int = 512,
           num_modalities: int = 3,
           output_dim: int = 512,
           dropout: float = 0.3)
  
  ARCHITECTURE:
    self.fusion_mlp = nn.Sequential(
        nn.Linear(512 * 3, 1024),      # 1536 → 1024
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),           # 1024 → 512
        nn.LayerNorm(512),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
  
  forward(self, video_feat, audio_feat, text_feat) -> Tensor:
    # Xử lý missing modalities: thay bằng zero vector
    features = []
    for feat in [video_feat, audio_feat, text_feat]:
        if feat is None:
            feat = torch.zeros(video_feat.shape[0], 512, device=video_feat.device)
        features.append(feat)
    
    x = torch.cat(features, dim=-1)    # (B, 1536)
    x = self.fusion_mlp(x)            # (B, 512)
    return x

═══════════════════════════════════════
GĐ2: CrossAttentionFusion (mạnh hơn)
═══════════════════════════════════════

CLASS: CrossAttentionFusion(nn.Module)
  __init__(self,
           input_dim: int = 512,
           num_heads: int = 8,
           num_layers: int = 2,
           dropout: float = 0.1)
  
  ARCHITECTURE:
    # Learnable modality tokens
    self.modality_embeddings = nn.Embedding(3, 512)  # 3 modalities
    
    # Transformer encoder for cross-modal interaction
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        batch_first=True
    )
    self.cross_attention = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    # Modality gating: học modality nào quan trọng
    self.gate = nn.Sequential(
        nn.Linear(512 * 3, 3),
        nn.Softmax(dim=-1)
    )
    
    self.output_projection = nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512)
    )
  
  forward(self, video_feat, audio_feat, text_feat) -> Tensor:
    B = video_feat.shape[0]
    
    # 1. Add modality embeddings
    mod_emb = self.modality_embeddings(torch.arange(3, device=video_feat.device))
    video_feat = video_feat + mod_emb[0]
    audio_feat = audio_feat + mod_emb[1]
    text_feat  = text_feat  + mod_emb[2]
    
    # 2. Stack as sequence: (B, 3, 512)
    x = torch.stack([video_feat, audio_feat, text_feat], dim=1)
    
    # 3. Cross-attention between modalities
    x = self.cross_attention(x)        # (B, 3, 512)
    
    # 4. Gating mechanism
    gate_input = torch.cat([video_feat, audio_feat, text_feat], dim=-1)
    gates = self.gate(gate_input)      # (B, 3)
    gates = gates.unsqueeze(-1)        # (B, 3, 1)
    x = (x * gates).sum(dim=1)        # (B, 512)
    
    # 5. Output projection
    x = self.output_projection(x)      # (B, 512)
    return x

═══════════════════════════════════════
Factory Function
═══════════════════════════════════════

def create_fusion(method: str = "concat", **kwargs) -> nn.Module:
    if method == "concat":
        return ConcatFusion(**kwargs)
    elif method == "cross_attention":
        return CrossAttentionFusion(**kwargs)
    else:
        raise ValueError(f"Unknown fusion method: {method}")
```

---

## 8. MULTIMODAL CLASSIFIER

### File: `models/classifier.py`

```
PURPOSE: Wrapper tổng hợp — quản lý 3 modules + fusion + classification head
INPUT:  Dict chứa video/audio/text data
OUTPUT: Dict {logits, probabilities, predicted_class, features}

CLASS: MultimodalClassifier(nn.Module)
  __init__(self,
           num_classes: int = 7,
           feature_dim: int = 512,
           fusion_method: str = "concat",     # "concat" hoặc "cross_attention"
           use_video: bool = True,
           use_audio: bool = True,
           use_text: bool = True,
           dropout: float = 0.3,
           **module_kwargs)
  
  ARCHITECTURE:
    # Modules (conditional)
    self.video_module = VideoModule(**kwargs) if use_video else None
    self.audio_module = AudioModule(**kwargs) if use_audio else None
    self.text_module  = TextModule(**kwargs)  if use_text  else None
    
    # Count active modalities
    self.num_active = sum([use_video, use_audio, use_text])
    
    # Fusion
    self.fusion = create_fusion(
        method=fusion_method,
        input_dim=feature_dim,
        num_modalities=self.num_active
    )
    
    # Classification head
    self.classifier = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes)
    )
  
  forward(self, batch: dict) -> dict:
    """
    batch keys:
      - "video_frames": (B, T, 3, 224, 224) — nếu use_video
      - "audio_features": (B, 120, time_steps) — nếu use_audio
      - "input_ids": (B, seq_len) — nếu use_text
      - "attention_mask": (B, seq_len) — nếu use_text
    """
    video_feat = self.video_module(batch["video_frames"]) if self.video_module else None
    audio_feat = self.audio_module(batch["audio_features"]) if self.audio_module else None
    text_feat  = self.text_module(batch["input_ids"], batch["attention_mask"]) if self.text_module else None
    
    # Fusion
    fused = self.fusion(video_feat, audio_feat, text_feat)  # (B, 512)
    
    # Classify
    logits = self.classifier(fused)                          # (B, num_classes)
    probs = F.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    
    return {
        "logits": logits,
        "probabilities": probs,
        "predicted_class": preds,
        "fused_features": fused,       # Để dùng cho visualization
        "video_features": video_feat,
        "audio_features": audio_feat,
        "text_features": text_feat,
    }
  
  get_num_parameters(self) -> dict:
    """Return số params cho mỗi module và tổng."""
    ...
  
  freeze_backbones(self):
    """Freeze tất cả pretrained backbones (ResNet, PhoBERT)."""
    ...
  
  unfreeze_backbones(self, last_n_layers: int = 2):
    """Unfreeze N layers cuối của backbones cho fine-tuning."""
    ...
```

---

## 9. DATASET & DATALOADER

### File: `training/dataset.py`

```
PURPOSE: Custom PyTorch Dataset — load và preprocess dữ liệu đa phương thức

CLASS: MultimodalEmotionDataset(torch.utils.data.Dataset)
  __init__(self,
           split_csv: str,                  # Path to train.csv / val.csv / test.csv
           video_frames_dir: str,
           audio_dir: str,
           transcripts_dir: str,
           tokenizer,                       # PhoBERT tokenizer
           num_frames: int = 16,
           max_audio_len: int = 300,
           max_text_len: int = 128,
           video_transform = None,
           augment_audio: bool = False)
  
  CSV FORMAT (split files):
    clip_id | emotion_label | emotion_index | split
    clip_001| happy         | 0             | train
    clip_002| sad           | 1             | train
  
  __len__(self) -> int
  
  __getitem__(self, idx) -> dict:
    Return:
    {
      "video_frames": Tensor(num_frames, 3, 224, 224),   # Padded/sampled
      "audio_features": Tensor(120, max_audio_len),       # MFCC + deltas
      "input_ids": Tensor(max_text_len),                  # PhoBERT tokens
      "attention_mask": Tensor(max_text_len),
      "label": int,                                        # 0-6
      "clip_id": str
    }
  
  INTERNAL METHODS:
    _load_video_frames(clip_id) -> Tensor
      - Load frames từ data/processed/video_frames/{clip_id}/
      - Uniform sampling: chọn num_frames frames cách đều
      - Apply transform (augmentation for train)
      - Pad nếu thiếu frames (repeat last frame)
    
    _load_audio_features(clip_id) -> Tensor
      - Load wav từ data/processed/audio/{clip_id}.wav
      - librosa.feature.mfcc → 40 features
      - Tính delta, delta-delta → 120 features
      - Pad/truncate → max_audio_len
      - Z-score normalize
    
    _load_text(clip_id) -> dict
      - Load transcript từ data/processed/transcripts/{clip_id}.json
      - Tokenize bằng PhoBERT tokenizer
      - Return {input_ids, attention_mask}

AUDIO AUGMENTATION (khi augment_audio=True):
  - Time stretching (0.8x - 1.2x)
  - Pitch shifting (±2 semitones)
  - Add Gaussian noise (SNR 15-30 dB)
  - Random volume change (0.7x - 1.3x)
```

### File: `training/dataloader.py`

```
PURPOSE: Factory cho DataLoader + collate function

FUNCTION: create_dataloaders(config: ProjectConfig) -> dict
  Return: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
  
  - train: shuffle=True, drop_last=True, num_workers=4, pin_memory=True
  - val/test: shuffle=False, drop_last=False
  - batch_size from config

FUNCTION: multimodal_collate_fn(batch: List[dict]) -> dict
  - Stack all tensors properly
  - Handle variable-length sequences (padding)
```

---

## 10. TRAINING PIPELINE

### File: `training/trainer.py`

```
PURPOSE: Training loop hoàn chỉnh với logging và checkpointing

CLASS: Trainer
  __init__(self,
           model: MultimodalClassifier,
           train_loader: DataLoader,
           val_loader: DataLoader,
           config: TrainingConfig,
           device: str = "cuda")
  
  METHODS:
    train(num_epochs: int) -> dict
      - Full training loop
      - Per-epoch: train_one_epoch() → evaluate() → log → checkpoint
      - Mixed precision (torch.cuda.amp)
      - Gradient clipping (max_norm=1.0)
      - Early stopping
      - Return best metrics
    
    train_one_epoch(epoch: int) -> dict
      - Iterate batches
      - Forward → loss → backward → step
      - Log every config.log_interval steps
      - Return {loss, accuracy}
    
    evaluate(loader: DataLoader) -> dict
      - No gradient computation
      - Return {loss, accuracy, f1_macro, f1_weighted, per_class_accuracy, confusion_matrix}
    
    save_checkpoint(epoch, metrics, is_best: bool)
      - Save to checkpoints/{model_name}_epoch{N}.pt
      - Save best model separately: checkpoints/best_model.pt
      - Include: model_state_dict, optimizer_state_dict, epoch, metrics, config
    
    load_checkpoint(path: str)
      - Resume training from checkpoint

OPTIMIZER: AdamW
  - lr=1e-4 (backbones), lr=1e-3 (heads and fusion)
  - Dùng parameter groups để set learning rate khác nhau:
    [
      {"params": backbone_params, "lr": 1e-5},
      {"params": fusion_params, "lr": 5e-4},
      {"params": classifier_params, "lr": 1e-3}
    ]

SCHEDULER: CosineAnnealingWarmRestarts
  - T_0=10, T_mult=2
  - Warmup 500 steps (linear warmup)

LOSS: CrossEntropyLoss with class weights
  - Tính class weights từ training set distribution (inverse frequency)
```

### File: `training/losses.py`

```
PURPOSE: Loss functions cho cả 2 giai đoạn

CLASS: EmotionLoss(nn.Module)
  """GĐ1: Weighted CrossEntropy"""
  __init__(self, class_weights: Tensor = None, label_smoothing: float = 0.1)
  forward(logits, labels) -> Tensor

CLASS: FocalLoss(nn.Module)
  """Xử lý class imbalance — focus vào samples khó"""
  __init__(self, alpha: float = 0.25, gamma: float = 2.0)
  forward(logits, labels) -> Tensor

CLASS: IncongruityLoss(nn.Module)
  """GĐ2: Đo sự mâu thuẫn giữa các modality predictions"""
  __init__(self, temperature: float = 0.5)
  forward(video_feat, audio_feat, text_feat) -> Tensor
    # Tính cosine similarity giữa các cặp modality
    # Penalize khi 2 modalities agree nhưng modality thứ 3 disagree mạnh
    # → Mô hình học nhận diện sarcasm/contradiction

CLASS: MultiTaskLoss(nn.Module)
  """GĐ2: Tổng hợp nhiều loss"""
  __init__(self, 
           alpha_cls: float = 1.0,          # Classification loss weight
           alpha_incongruity: float = 0.3,   # Incongruity loss weight
           alpha_diversity: float = 0.1)     # Feature diversity loss weight
  forward(outputs: dict, labels: Tensor) -> dict
    Return: {"total_loss": ..., "cls_loss": ..., "incongruity_loss": ..., "diversity_loss": ...}
```

### File: `training/evaluate.py`

```
PURPOSE: Evaluation metrics + visualization

FUNCTION: evaluate_model(model, test_loader, device, emotion_labels) -> dict
  Return:
  {
    "accuracy": float,
    "f1_macro": float,
    "f1_weighted": float,
    "per_class": {
      "happy": {"precision": .., "recall": .., "f1": ..},
      ...
    },
    "confusion_matrix": np.ndarray (7x7)
  }

FUNCTION: plot_confusion_matrix(cm, labels, save_path) -> None
  - Seaborn heatmap
  - Save PNG

FUNCTION: plot_training_history(history: dict, save_path) -> None
  - Loss curve, accuracy curve
  - Save PNG

FUNCTION: compare_models(results: dict) -> pd.DataFrame
  - So sánh: unimodal (V, A, T) vs bimodal (VA, VT, AT) vs trimodal (VAT)
  - Export bảng đẹp cho báo cáo
```

### File: `training/utils.py`

```
PURPOSE: Utility functions

FUNCTION: set_seed(seed: int = 42) → None
FUNCTION: get_device() → torch.device
FUNCTION: count_parameters(model) → dict
FUNCTION: EarlyStopping class
FUNCTION: save_checkpoint(state, filepath)
FUNCTION: load_checkpoint(filepath) → dict
```

---

## 11. BACKEND API

### File: `backend/app.py`

```
PURPOSE: Flask application factory

FUNCTION: create_app(config_name: str = "default") -> Flask
  - Initialize Flask app
  - Register blueprints (api routes)
  - Setup CORS (allow all origins for dev)
  - Configure upload folder, max file size (200MB)
  - Load model on startup
  - Serve frontend static files

RUN:
  if __name__ == "__main__":
      app = create_app()
      app.run(host="0.0.0.0", port=5000, debug=True)
```

### File: `backend/api/routes.py`

```
PURPOSE: Tất cả API endpoints

ENDPOINTS:

POST /api/predict
  - Input: multipart/form-data với field "video" (file .mp4/.avi/.mov)
  - Process: preprocess video → run model → return results
  - Response (200):
    {
      "success": true,
      "prediction": {
        "emotion": "happy",
        "emotion_vi": "Vui vẻ",
        "confidence": 0.87,
        "all_scores": {
          "happy": 0.87,
          "sad": 0.03,
          "angry": 0.02,
          "fear": 0.01,
          "surprise": 0.04,
          "disgust": 0.01,
          "neutral": 0.02
        }
      },
      "analysis": {
        "video_dominant": "happy",
        "audio_dominant": "happy",
        "text_dominant": "neutral",
        "transcript": "Anh ơi, em vui quá!",
        "modality_agreement": 0.78,
        "processing_time_ms": 1250
      }
    }
  - Response (400): {"success": false, "error": "Invalid file format"}
  - Response (500): {"success": false, "error": "Internal processing error"}

POST /api/predict/text
  - Input: JSON {"text": "Tôi rất buồn hôm nay"}
  - Response: Same structure, only text analysis

GET /api/health
  - Response: {"status": "ok", "model_loaded": true, "gpu_available": true}

GET /api/model/info
  - Response: {
      "model_name": "MultimodalClassifier",
      "fusion_method": "cross_attention",
      "num_parameters": 28500000,
      "num_classes": 7,
      "modalities": ["video", "audio", "text"],
      "checkpoint": "best_model_epoch_42.pt"
    }
```

### File: `backend/services/preprocessor.py`

```
PURPOSE: Preprocessing video cho inference

CLASS: VideoPreprocessor
  __init__(self, config: ProjectConfig)
  
  METHODS:
    process_video(video_path: Path) -> dict
      - Extract frames → MTCNN crop faces → transform → tensor
      - Extract audio → MFCC features → tensor
      - Whisper STT → PhoBERT tokenize → tensor
      - Return dict ready for model.forward()
    
    process_text_only(text: str) -> dict
      - Tokenize → tensor
      - Return dict for text-only inference
```

### File: `backend/services/predictor.py`

```
PURPOSE: Model loading và inference

CLASS: EmotionPredictor
  __init__(self, checkpoint_path: str, device: str = "cuda")
  
  METHODS:
    load_model(checkpoint_path) -> None
    predict(preprocessed_data: dict) -> dict
    predict_video(video_path: Path) -> dict   # End-to-end
    predict_text(text: str) -> dict           # Text only
```

---

## 12. FRONTEND UI

### File: `frontend/index.html`

```
PURPOSE: Single-page application cho phân tích cảm xúc

LAYOUT:
  ┌────────────────────────────────────────────────────┐
  │  HEADER: Logo + "Phân tích Cảm xúc Đa phương thức"│
  ├────────────────────────────────────────────────────┤
  │                                                    │
  │  ┌──────────────────────────────────────────────┐  │
  │  │                                              │  │
  │  │          UPLOAD ZONE                         │  │
  │  │          (Drag & Drop Video)                 │  │
  │  │          Hoặc chọn file                      │  │
  │  │                                              │  │
  │  │    [📁 Chọn Video]  [📝 Nhập Text]          │  │
  │  │                                              │  │
  │  └──────────────────────────────────────────────┘  │
  │                                                    │
  │  ┌─────────────┐  ┌────────────────────────────┐  │
  │  │             │  │                            │  │
  │  │  VIDEO      │  │  KẾT QUẢ PHÂN TÍCH        │  │
  │  │  PREVIEW    │  │                            │  │
  │  │             │  │  Cảm xúc: 😊 Vui vẻ       │  │
  │  │  (player)   │  │  Độ tin cậy: 87%          │  │
  │  │             │  │                            │  │
  │  │             │  │  ┌────────────────────┐    │  │
  │  │             │  │  │ BAR CHART          │    │  │
  │  │             │  │  │ (7 emotions)       │    │  │
  │  │             │  │  └────────────────────┘    │  │
  │  │             │  │                            │  │
  │  └─────────────┘  │  Phân tích chi tiết:       │  │
  │                    │  🎬 Video: Vui vẻ          │  │
  │                    │  🔊 Audio: Vui vẻ          │  │
  │                    │  📝 Text: Trung tính       │  │
  │                    │                            │  │
  │                    │  Lời thoại: "Anh ơi..."    │  │
  │                    └────────────────────────────┘  │
  │                                                    │
  ├────────────────────────────────────────────────────┤
  │  FOOTER: Đồ án Deep Learning 2025-2026            │
  └────────────────────────────────────────────────────┘

DESIGN SPECS:
  - Dark theme: background #0a0a0f, card background rgba(255,255,255,0.05)
  - Glassmorphism: backdrop-filter: blur(20px)
  - Accent color: gradient from #667eea to #764ba2
  - Font: Inter (Google Fonts)
  - Animations: fadeIn, slideUp on results
  - Responsive: flexbox/grid, mobile-friendly
  - Loading: skeleton animation + progress bar during upload/inference

DEPENDENCIES (CDN):
  - Chart.js 4.4+
  - Google Fonts: Inter
  - Không cần React/Vue — vanilla JS đủ cho SPA này
```

### File: `frontend/js/app.js`

```
PURPOSE: Main controller — quản lý state và UI

MODULES:
  - initApp(): Setup event listeners, initialize UI
  - handleFileSelect(file): Validate file, preview video, enable analyze button
  - handleDrop(event): Drag & drop handler
  - handleTextMode(): Switch sang chế độ nhập text
  - analyzeVideo(file): Call API → display results
  - analyzeText(text): Call API text endpoint → display results
  - displayResults(data): Render emotion result + chart + details
  - showLoading(): Show loading skeleton
  - hideLoading(): Hide loading
  - showError(message): Display error toast
```

### File: `frontend/js/api.js`

```
PURPOSE: Backend API communication

FUNCTIONS:
  async predictVideo(file: File) -> Object
    - POST /api/predict with FormData
    - Handle progress events
  
  async predictText(text: string) -> Object
    - POST /api/predict/text with JSON
  
  async checkHealth() -> Object
    - GET /api/health
  
  async getModelInfo() -> Object
    - GET /api/model/info
```

### File: `frontend/js/charts.js`

```
PURPOSE: Chart.js visualizations

FUNCTIONS:
  createEmotionBarChart(canvasId, scores) -> Chart
    - Horizontal bar chart
    - 7 emotions with color coding
    - Animated entrance
  
  createRadarChart(canvasId, scores) -> Chart
    - Radar/spider chart cho overall emotion profile
  
  updateChart(chart, newData) -> void
```

---

## 13. DATA SCHEMA & FORMATS

### 13.1 clips_manifest.csv
```csv
clip_id,source_video,movie_name,start_time,end_time,duration,clip_path
clip_0001,ep01.mp4,ve_nha_di_con,00:05:23,00:05:31,8.0,data/processed/clips/clip_0001.mp4
```

### 13.2 annotations.csv (final labels)
```csv
clip_id,emotion_label,emotion_index,annotator,confidence,notes
clip_0001,happy,0,human,1.0,
clip_0002,sad,1,human,1.0,đoạn khóc rõ ràng
clip_0003,angry,2,deepface_verified,0.85,AI đoán đúng
```

### 13.3 splits/{train|val|test}.csv
```csv
clip_id,emotion_label,emotion_index
clip_0001,happy,0
clip_0002,sad,1
```

### 13.4 transcripts/{clip_id}.json
```json
{
  "clip_id": "clip_0001",
  "text": "Anh ơi, em vui quá! Cuối cùng cũng đậu rồi!",
  "segments": [
    {"start": 0.0, "end": 1.5, "text": "Anh ơi,"},
    {"start": 1.5, "end": 3.2, "text": "em vui quá!"},
    {"start": 3.5, "end": 5.0, "text": "Cuối cùng cũng đậu rồi!"}
  ],
  "language": "vi",
  "model": "whisper-medium",
  "confidence": 0.94
}
```

### 13.5 Model Checkpoint Format
```python
{
    "epoch": 42,
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "best_val_accuracy": 0.723,
    "best_val_f1": 0.698,
    "config": ProjectConfig,  # Lưu config để reproduce
    "training_history": {
        "train_loss": [...],
        "val_loss": [...],
        "val_accuracy": [...],
        "val_f1": [...]
    }
}
```

---

## 14. CHIẾN LƯỢC 2 GIAI ĐOẠN

### Giai đoạn 1: Nền tảng

| # | Module/File | Ưu tiên | Phụ thuộc | Mục tiêu |
|---|---|---|---|---|
| 1 | `training/config.py` | ★★★ | — | ✅ Đã tạo |
| 2 | `tools/*` (data pipeline) | ★★★ | config | Dataset sẵn sàng |
| 3 | `models/video_module.py` | ★★★ | — | >55% acc standalone |
| 4 | `models/audio_module.py` | ★★★ | — | >45% acc standalone |
| 5 | `models/text_module.py` | ★★★ | — | >60% acc standalone |
| 6 | `models/fusion.py` (Concat) | ★★★ | 3,4,5 | — |
| 7 | `models/classifier.py` | ★★★ | 3,4,5,6 | >65% acc combined |
| 8 | `training/dataset.py` | ★★★ | 2 | — |
| 9 | `training/trainer.py` | ★★★ | 7,8 | End-to-end train |
| 10 | `training/losses.py` | ★★ | — | CE + Focal |
| 11 | `training/evaluate.py` | ★★ | 7 | Metrics + charts |
| 12 | `backend/*` | ★★ | 7 | API serving |
| 13 | `frontend/*` | ★★ | 12 | Web UI |

### Giai đoạn 2: Nâng cấp

| # | Nâng cấp | File ảnh hưởng | Mục tiêu |
|---|---|---|---|
| 1 | CrossAttentionFusion | `models/fusion.py` | >70% acc |
| 2 | IncongruityLoss | `training/losses.py` | Detect sarcasm |
| 3 | Noise Injection | `training/trainer.py`, `training/dataset.py` | Robust to noise |
| 4 | PhoBERT unfreeze | `models/text_module.py` | Better text features |
| 5 | Prototype Layer | `models/classifier.py` (new) | Explainability |

---

## 15. TESTING & VERIFICATION

### Unit Test Specs

```python
# tests/test_video_module.py
def test_video_module_output_shape():
    model = VideoModule()
    x = torch.randn(2, 16, 3, 224, 224)  # batch=2, 16 frames
    out = model(x)
    assert out.shape == (2, 512)

def test_video_module_single_frame():
    model = VideoModule(num_frames=1)
    x = torch.randn(1, 1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 512)

# tests/test_audio_module.py
def test_audio_module_output_shape():
    model = AudioModule()
    x = torch.randn(2, 120, 300)  # batch=2, 120 features, 300 time steps
    out = model(x)
    assert out.shape == (2, 512)

# tests/test_text_module.py
def test_text_module_output_shape():
    model = TextModule()
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    out = model(input_ids, attention_mask)
    assert out.shape == (2, 512)

# tests/test_fusion.py
def test_concat_fusion():
    fusion = ConcatFusion()
    v = torch.randn(2, 512)
    a = torch.randn(2, 512)
    t = torch.randn(2, 512)
    out = fusion(v, a, t)
    assert out.shape == (2, 512)

def test_fusion_missing_modality():
    fusion = ConcatFusion()
    v = torch.randn(2, 512)
    out = fusion(v, None, None)  # Only video
    assert out.shape == (2, 512)

# tests/test_classifier.py
def test_full_pipeline():
    model = MultimodalClassifier(num_classes=7)
    batch = {
        "video_frames": torch.randn(2, 16, 3, 224, 224),
        "audio_features": torch.randn(2, 120, 300),
        "input_ids": torch.randint(0, 1000, (2, 128)),
        "attention_mask": torch.ones(2, 128),
    }
    output = model(batch)
    assert output["logits"].shape == (2, 7)
    assert output["probabilities"].shape == (2, 7)
    assert output["predicted_class"].shape == (2,)
```

### Performance Targets

| Configuration | Accuracy | F1 (macro) | Note |
|---|---|---|---|
| Video only | >55% | >50% | 7 classes baseline |
| Audio only | >45% | >40% | Hardest standalone |
| Text only | >60% | >55% | PhoBERT helps |
| V + A | >60% | >55% | |
| V + T | >65% | >60% | |
| A + T | >58% | >53% | |
| V + A + T (GĐ1) | >68% | >63% | ConcatFusion |
| V + A + T (GĐ2) | >73% | >68% | CrossAttention |

---

## 16. DEPENDENCY MAP

```
Thứ tự implement (từ trên xuống):

training/config.py          ← ĐÃ CÓ
        │
tools/* (data pipeline)     ← CẦN LÀM ĐẦU TIÊN
        │
        ├── models/video_module.py     ← Độc lập, làm song song được
        ├── models/audio_module.py     ← Độc lập
        └── models/text_module.py      ← Độc lập
                │
        models/fusion.py               ← Phụ thuộc 3 modules trên
                │
        models/classifier.py           ← Phụ thuộc fusion
                │
        ├── training/dataset.py        ← Phụ thuộc data pipeline
        ├── training/losses.py         ← Độc lập
        └── training/utils.py          ← Độc lập
                │
        training/trainer.py            ← Phụ thuộc dataset, model, losses
                │
        training/evaluate.py           ← Phụ thuộc model
                │
        ├── backend/services/*         ← Phụ thuộc model
        ├── backend/api/*              ← Phụ thuộc services
        └── backend/app.py             ← Phụ thuộc api
                │
        frontend/*                     ← Phụ thuộc backend API
                │
        tests/*                        ← Cuối cùng (hoặc TDD)
```

---

> **CHÚ Ý CHO AI AGENT:** Khi implement bất kỳ file nào, luôn:
> 1. Đọc `training/config.py` để dùng đúng config values
> 2. Đảm bảo output tensor shape khớp với spec ở trên (tất cả modules output `(B, 512)`)
> 3. Xử lý missing modalities (None) trong fusion
> 4. Sử dụng type hints cho tất cả function signatures
> 5. Viết docstring cho tất cả classes và public methods
> 6. Import paths luôn bắt đầu từ project root: `from models.video_module import VideoModule`
