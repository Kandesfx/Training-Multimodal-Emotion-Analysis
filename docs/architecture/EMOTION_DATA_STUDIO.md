# 🎬 EMOTION DATA STUDIO — Kiến trúc Phần mềm v2.0
## Phần mềm Desktop Native (PySide6) Khai thác & Quản lý Dataset Cảm xúc Đa phương thức

> **Đây là phần mềm riêng biệt**, nằm tại `tools/emotion-data-studio/`, tách khỏi pipeline training chính. Mục tiêu: **tự động hóa tối đa** việc thu thập, xử lý, gán nhãn, kiểm duyệt và xuất dataset cảm xúc từ video tiếng Việt.

---

## 1. TỔNG QUAN SẢN PHẨM

### 1.1 Mô tả

**Emotion Data Studio (EDS)** là ứng dụng **desktop native** dựa trên PySide6 (Qt6), cho phép:
- Tải video từ YouTube → tự động cắt scene → tự động detect khuôn mặt → tự động gán nhãn cảm xúc → hiển thị trực quan cho người kiểm duyệt → xuất dataset

Người dùng chỉ cần: **nhập URL → chờ xử lý → kiểm duyệt → xuất dữ liệu**.

### 1.2 Thay đổi kiến trúc (v1 → v2)

| Tiêu chí | v1 (Electron + FastAPI) | v2 (PySide6) |
|---|---|---|
| Kiến trúc | Web app bọc desktop shell | **Desktop native thuần** |
| Kích thước | ~200MB+ (bundled Chromium) | **~30-50MB** |
| Số process | 2 (Electron + FastAPI server) | **1 duy nhất** |
| Giao tiếp UI↔Backend | HTTP REST API | **Gọi hàm Python trực tiếp** |
| Phụ thuộc | Node.js + Python | **Chỉ Python** |
| Đóng gói | electron-builder | **PyInstaller + Inno Setup** |
| Background tasks | Celery + Redis | **QThread (native)** |

### 1.3 Sơ đồ Tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│              EMOTION DATA STUDIO (.exe)                           │
│              Single Python Process                                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    UI LAYER (PySide6 / Qt6)                  │  │
│  │                                                              │  │
│  │  MainWindow                                                  │  │
│  │  ├── Sidebar Navigation (4 pages)                           │  │
│  │  ├── DashboardPage     — Thống kê + Import video            │  │
│  │  ├── ProcessingPage    — Monitor pipeline real-time          │  │
│  │  ├── ReviewPage        — Video player + AI prediction       │  │
│  │  │   └── QMediaPlayer (Windows Media Foundation)            │  │
│  │  └── ExportPage        — Export dataset + Cloud sync        │  │
│  │                                                              │  │
│  │  Styling: QSS (Qt Style Sheets) — Dark theme                │  │
│  │  Threading: QThread workers (non-blocking UI)                │  │
│  └──────────────────────┬──────────────────────────────────────┘  │
│                          │ Direct Python calls (no HTTP)          │
│  ┌──────────────────────▼──────────────────────────────────────┐  │
│  │               BACKEND LAYER (Python services)                │  │
│  │                                                              │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐   │  │
│  │  │ Database    │ │ AI Pipeline  │ │ Cloud Sync          │   │  │
│  │  │ (SQLAlchemy │ │ 7 Services   │ │ (GCS + CloudSQL)    │   │  │
│  │  │  + SQLite)  │ │              │ │                     │   │  │
│  │  └─────────────┘ └──────────────┘ └─────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    LOCAL STORAGE                               │  │
│  │  SQLite DB │ File System (videos, clips, frames, audio)      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. CẤU TRÚC THƯ MỤC

```
BCDA/                                    # Đồ án chính
├── training/                            # Code train model
├── models/                              # Kiến trúc model
├── notebooks/                           # Jupyter notebooks
├── checkpoints/                         # Model weights
├── docs/                                # Tài liệu
│
└── tools/
    └── emotion-data-studio/             # ★ Tool EDS
        ├── app.py                       # Desktop entry point
        ├── app_cloud.py                 # Cloud Run entry point
        ├── requirements.txt             # EDS dependencies
        ├── requirements-cloud.txt       # Cloud-only dependencies
        ├── README.md
        ├── .gitignore
        │
        ├── backend/                     # Backend services
        │   ├── __init__.py
        │   ├── config.py               # Centralized config (Pydantic)
        │   ├── database/
        │   │   ├── local_db.py          # SQLite engine + session
        │   │   └── models.py           # Video, Clip, Label, SyncLog
        │   ├── services/
        │   │   ├── downloader.py       # yt-dlp video download
        │   │   ├── scene_splitter.py   # PySceneDetect
        │   │   ├── face_extractor.py   # SCRFD + ByteTrack
        │   │   ├── audio_extractor.py  # FFmpeg audio extraction
        │   │   ├── transcriber.py      # OpenAI Whisper (vi)
        │   │   ├── emotion_analyzer.py # Ensemble voting (4 models)
        │   │   ├── quality_scorer.py   # Quality scoring + routing
        │   │   └── pipeline_orchestrator.py  # Pipeline controller
        │   ├── ai_models/
        │   │   └── model_manager.py    # Lazy loading + GPU mgmt
        │   ├── api/                     # FastAPI routers (Cloud Run only)
        │   │   ├── videos.py
        │   │   ├── clips.py
        │   │   └── labels.py
        │   └── cloud/
        │       ├── gcs_client.py       # Google Cloud Storage
        │       ├── cloudsql_client.py  # Cloud SQL (PostgreSQL)
        │       └── sync_manager.py     # Bi-directional sync
        │
        ├── ui/                          # PySide6 Desktop UI
        │   ├── main_window.py          # MainWindow + sidebar + statusbar
        │   ├── updater.py             # Auto-update (Cloudflare R2)
        │   ├── pages/
        │   │   ├── dashboard_page.py   # Stats + Import + Video table
        │   │   ├── processing_page.py  # Pipeline monitor + live log
        │   │   ├── review_page.py     # Video player + labeling studio
        │   │   └── export_page.py     # Export + cloud sync
        │   ├── widgets/
        │   │   └── sidebar.py         # Navigation sidebar
        │   ├── workers/
        │   │   ├── pipeline_worker.py  # QThread: AI pipeline
        │   │   └── export_worker.py   # QThread: dataset export
        │   └── styles/
        │       ├── theme.py           # Design tokens (colors, spacing)
        │       └── dark_theme.qss     # Qt stylesheet (18KB)
        │
        ├── build/                       # Build tools
        │   ├── emotion_studio.spec     # PyInstaller spec
        │   └── build.ps1              # Build automation script
        │
        ├── deploy/                      # Cloud deployment
        │   ├── Dockerfile             # Cloud Run container
        │   └── cloudbuild.yaml        # Cloud Build CI/CD
        │
        ├── installer/                   # Windows installer
        │   └── emotion_data_studio.iss # Inno Setup script
        │
        └── data/                        # Local data (gitignored)
            ├── videos/
            ├── clips/
            ├── frames/
            ├── audio/
            ├── transcripts/
            ├── exports/
            ├── models_cache/
            └── studio.db              # SQLite database
```

---

## 3. AI PIPELINE

### 3.1 Luồng xử lý (7 stages)

```
URL/File Input
     │
     ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ 1. Download │───▶│ 2. Scene     │───▶│ 3. Face         │
│    (yt-dlp) │    │    Split     │    │    Detection    │
│             │    │ (PySceneDetect)   │ (SCRFD+ByteTrack)│
└─────────────┘    └──────────────┘    └────────┬────────┘
                                                │
     ┌──────────────────────────────────────────┘
     │
     ▼
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│ 4. Audio        │───▶│ 5. Transcribe│───▶│ 6. Emotion      │
│    Extract      │    │    (Whisper) │    │    Analyze       │
│    (FFmpeg)     │    │    Vietnamese│    │    (Ensemble)    │
└─────────────────┘    └──────────────┘    └────────┬────────┘
                                                    │
                                                    ▼
                                          ┌──────────────────┐
                                          │ 7. Quality Score │
                                          │    + Auto-route  │
                                          └──────────────────┘
```

### 3.2 Ensemble Emotion Analyzer

4 models vote cho cảm xúc cuối cùng:

| Model | Modality | Input | Output |
|---|---|---|---|
| **HSEmotion** | Visual | Face crops | 7 emotions + confidence |
| **DeepFace** | Visual | Face crops | 7 emotions + confidence |
| **PhoBERT** | Text | Vietnamese transcript | sentiment polarity |
| **Wav2Vec2** | Audio | Audio MFCC | emotion from voice |

**Voting mechanism**: Weighted average → top emotion + agreement score + incongruity flag.

### 3.3 Quality Scoring

```
quality_score = w1 * face_confidence
             + w2 * audio_quality
             + w3 * transcript_confidence
             + w4 * model_agreement
             + w5 * clip_duration_factor
```

Routing rules:
- Score ≥ 0.8 + agreement ≥ 3/4 → `auto_approved`
- Score ≥ 0.5 → `needs_review`
- Score < 0.5 → `pending` (manual review required)

---

## 4. DATABASE SCHEMA

### 4.1 SQLite Tables

```sql
-- Video metadata
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    movie_name TEXT,
    source_url TEXT,
    file_path TEXT,
    gcs_path TEXT,
    duration_sec REAL,
    resolution TEXT,
    status TEXT DEFAULT 'pending',
    total_clips INTEGER DEFAULT 0,
    approved_clips INTEGER DEFAULT 0,
    created_at DATETIME,
    updated_at DATETIME
);

-- Processed clips
CREATE TABLE clips (
    id TEXT PRIMARY KEY,
    video_id TEXT REFERENCES videos(id),
    clip_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    clip_path TEXT,
    gcs_path TEXT,
    num_frames INTEGER DEFAULT 0,
    num_faces INTEGER DEFAULT 0,
    transcript TEXT,
    speaker_id TEXT,
    quality_score REAL DEFAULT 0.0,
    status TEXT DEFAULT 'pending',
    predicted_emotion TEXT,
    confidence REAL DEFAULT 0.0,
    agreement TEXT,
    has_incongruity BOOLEAN DEFAULT FALSE,
    all_scores JSON,
    per_model_scores JSON,
    user_emotion TEXT,
    reviewer_notes TEXT,
    reviewed_at DATETIME,
    created_at DATETIME,
    updated_at DATETIME
);

-- Final labels for training
CREATE TABLE labels (
    clip_id TEXT PRIMARY KEY REFERENCES clips(id),
    emotion_label TEXT NOT NULL,
    emotion_index INTEGER NOT NULL,
    label_source TEXT,
    confidence REAL,
    split TEXT,          -- train/val/test
    exported BOOLEAN DEFAULT FALSE,
    created_at DATETIME
);

-- Sync log
CREATE TABLE sync_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    direction TEXT,
    entity_type TEXT,
    entity_id TEXT,
    status TEXT,
    error_message TEXT,
    synced_at DATETIME
);
```

---

## 5. UI DESIGN

### 5.1 Design System

- **Theme**: Dark mode deep (#0a0a0f background, #6c5ce7 purple accent)
- **Font**: Inter (Google Fonts), JetBrains Mono (code/logs)
- **Cards**: Glassmorphism (rgba borders, subtle shadows)
- **Stylesheet**: 18KB QSS covering all Qt widgets

### 5.2 Pages

| Page | Chức năng | Keyboard Shortcuts |
|---|---|---|
| **Dashboard** | Stats cards, import form, video table | — |
| **Processing** | 7-stage progress, live log, GPU/RAM monitor | — |
| **Review Studio** | Video player, AI scores, emotion labeling | 1-7 (emotion), A (approve), R (reject), ←→ (nav), Space (play/pause) |
| **Export** | Format selection, train/val/test split, cloud sync | — |

### 5.3 Threading Model

```
Main Thread (Qt Event Loop)
├── UI rendering + user input
├── Signal/Slot connections
│
├── PipelineWorker (QThread)
│   └── Runs 7-stage AI pipeline
│       └── Emits: progress_updated, log_message, pipeline_finished
│
└── ExportWorker (QThread)
    └── Runs dataset export + split
        └── Emits: progress_updated, export_finished
```

---

## 6. CLOUD INTEGRATION

### 6.1 Architecture

```
Desktop App (SQLite)  ←→  Cloud SQL (PostgreSQL)
Local Files           ←→  Google Cloud Storage
                           │
                     Cloud Run (FastAPI API)
```

### 6.2 Sync Strategy

- **Metadata sync**: Timestamp-based comparison (newer wins)
- **File sync**: Upload only (local → GCS), skip existing files
- **Conflict resolution**: Last-write-wins with sync log

### 6.3 Cloud SQL Connection Methods

1. **Cloud SQL Python Connector** — recommended for Cloud Run
2. **Cloud SQL Auth Proxy** — for local development
3. **Direct TCP** — for VPC-connected services

---

## 7. PACKAGING & DISTRIBUTION

### 7.1 Build Pipeline

```
Source Code
    │
    ▼
PyInstaller (emotion_studio.spec)
    │ Excludes: torch, transformers, whisper (runtime loaded)
    │ Includes: PySide6, SQLAlchemy, UI modules
    ▼
dist/EmotionDataStudio/
    │
    ▼
Inno Setup (emotion_data_studio.iss)
    │ Adds: FFmpeg, data dirs, registry, env vars
    ▼
EmotionDataStudio-1.0.0-Setup.exe
    │
    ▼
Upload to Cloudflare R2 (auto-update)
```

### 7.2 Auto-Update Flow

```
App starts → check R2/latest.json → compare versions
   │
   ├── No update → continue
   └── Update available → show dialog → download .exe → install → restart
```
