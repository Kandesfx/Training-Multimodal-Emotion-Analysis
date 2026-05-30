# 📋 HƯỚNG DẪN SETUP & DEPLOY — Emotion Data Studio

> Tài liệu hướng dẫn đầy đủ cách thiết lập môi trường phát triển, chạy ứng dụng, đóng gói cài đặt và deploy lên Google Cloud.

---

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Cài đặt môi trường phát triển](#2-cài-đặt-môi-trường-phát-triển)
3. [Chạy ứng dụng Desktop](#3-chạy-ứng-dụng-desktop)
4. [Cấu hình .env](#4-cấu-hình-env)
5. [Đóng gói bằng PyInstaller](#5-đóng-gói-bằng-pyinstaller)
6. [Tạo Windows Installer (Inno Setup)](#6-tạo-windows-installer-inno-setup)
7. [Deploy Cloud Run (Google Cloud)](#7-deploy-cloud-run-google-cloud)
8. [Setup Auto-Updater (Cloudflare R2)](#8-setup-auto-updater-cloudflare-r2)
9. [Xử lý lỗi thường gặp](#9-xử-lý-lỗi-thường-gặp)

---

## 1. Yêu cầu hệ thống

### Hardware tối thiểu

| Component | Tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | Không bắt buộc | NVIDIA GTX 1060+ (CUDA 11.8+) |
| Disk | 10 GB free | 50+ GB (cho video data) |
| OS | Windows 10 64-bit | Windows 11 |

### Software yêu cầu

| Phần mềm | Phiên bản | Mục đích | Link tải |
|---|---|---|---|
| **Python** | 3.10 - 3.12 | Runtime | [python.org](https://python.org/downloads) |
| **FFmpeg** | 6.0+ | Xử lý video/audio | [ffmpeg.org](https://ffmpeg.org/download.html) |
| **Git** | 2.40+ | Version control | [git-scm.com](https://git-scm.com) |
| **CUDA Toolkit** | 11.8+ (tùy chọn) | GPU acceleration | [nvidia.com](https://developer.nvidia.com/cuda-toolkit) |

### Phần mềm đóng gói (tùy chọn)

| Phần mềm | Mục đích |
|---|---|
| **Inno Setup 6** | Tạo Windows installer |
| **Docker Desktop** | Build Cloud Run container |
| **gcloud CLI** | Deploy lên Google Cloud |

---

## 2. Cài đặt môi trường phát triển

### Bước 1: Clone repository

```bash
git clone https://github.com/your-team/BCDA.git
cd BCDA
```

### Bước 2: Tạo Virtual Environment

```bash
# Tạo venv
python -m venv venv

# Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Hoặc (Windows CMD)
.\venv\Scripts\activate.bat
```

### Bước 3: Cài đặt dependencies cho EDS tool

```bash
cd tools/emotion-data-studio

# Cài đặt dependencies cơ bản
pip install -r requirements.txt
```

### Bước 4: Cài đặt PyTorch (GPU — khuyến nghị)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only (chậm hơn nhưng vẫn chạy được)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Bước 5: Cài đặt FFmpeg

**Cách 1: Winget (Windows 11)**
```bash
winget install Gyan.FFmpeg
```

**Cách 2: Manual**
1. Tải từ [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/)
2. Giải nén vào `C:\ffmpeg\`
3. Thêm `C:\ffmpeg\bin` vào System PATH
4. Kiểm tra: `ffmpeg -version`

**Cách 3: Đặt trong project**
```
tools/emotion-data-studio/bin/
├── ffmpeg.exe
└── ffprobe.exe
```

### Bước 6: Kiểm tra cài đặt

```bash
cd tools/emotion-data-studio

# Test tất cả imports
python -c "
from backend.config import settings
print(f'Config OK: {settings.APP_NAME}')

from backend.database.local_db import init_database
init_database()
print('Database OK')

from ui.main_window import MainWindow
print('UI OK')

import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')

print('ALL CHECKS PASSED!')
"
```

---

## 3. Chạy ứng dụng Desktop

### Development mode

```bash
cd tools/emotion-data-studio
python app.py
```

Ứng dụng sẽ:
1. Khởi tạo database SQLite tại `data/studio.db`
2. Tải dark theme stylesheet
3. Mở cửa sổ MainWindow (1440×900)
4. Kiểm tra cập nhật tự động (silent)

### Giao diện chính

| Trang | Chức năng |
|---|---|
| **📊 Dashboard** | Thống kê tổng quan, nhập video URL, danh sách video |
| **⚙️ Processing** | Theo dõi pipeline AI real-time (7 stages), log live |
| **🎭 Review Studio** | Video player + AI predictions + gán nhãn cảm xúc |
| **📦 Export** | Xuất dataset, chọn format, train/val/test split |

### Keyboard Shortcuts (Review Studio)

| Phím | Hành động |
|---|---|
| `1-7` | Gán nhãn cảm xúc (happy, sad, angry, ...) |
| `A` | Approve clip |
| `R` | Reject clip |
| `←` `→` | Prev/Next clip |
| `Space` | Play/Pause video |

---

## 4. Cấu hình .env

Tạo file `.env` tại `tools/emotion-data-studio/.env`:

```env
# ============================================================
# Emotion Data Studio — Environment Configuration
# ============================================================

# === Chế độ chạy ===
ENV=development                    # development | production

# === Google Cloud (tùy chọn) ===
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GCP_PROJECT_ID=your-project-id
GCS_BUCKET_NAME=eds-data-bucket

# === Cloud SQL (tùy chọn) ===
CLOUD_SQL_CONNECTION_NAME=project:region:instance
CLOUD_SQL_USER=eds_user
CLOUD_SQL_PASSWORD=your-secure-password
CLOUD_SQL_DB=emotion_studio

# === Auto-Updater ===
EDS_UPDATE_URL=https://updates.your-domain.com/releases

# === Cloudflare R2 (cho release hosting) ===
R2_ENDPOINT=https://xxxx.r2.cloudflarestorage.com
AWS_ACCESS_KEY_ID=your-r2-access-key
AWS_SECRET_ACCESS_KEY=your-r2-secret-key

# === FFmpeg (nếu không nằm trong system PATH) ===
# EDS_FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
# EDS_FFPROBE_PATH=C:\ffmpeg\bin\ffprobe.exe

# === GPU ===
# USE_GPU=true
```

> ⚠️ **Quan trọng**: File `.env` đã được thêm vào `.gitignore` — **KHÔNG BAO GIỜ** commit file này lên GitHub!

---

## 5. Đóng gói bằng PyInstaller

### Cài đặt PyInstaller

```bash
pip install pyinstaller>=6.0
```

### Build nhanh

```bash
cd tools/emotion-data-studio

# Build bằng spec file
pyinstaller --noconfirm --clean build/emotion_studio.spec
```

### Build bằng script tự động

```powershell
# Windows PowerShell
cd tools/emotion-data-studio
.\build\build.ps1 -Version "1.0.0"

# Bỏ qua Inno Setup
.\build\build.ps1 -Version "1.0.0" -SkipInstaller

# Debug mode
.\build\build.ps1 -Version "1.0.0" -Debug
```

### Kết quả build

```
tools/emotion-data-studio/
├── dist/
│   └── EmotionDataStudio/           # App bundle
│       ├── EmotionDataStudio.exe    # ★ Chạy file này
│       ├── ui/styles/dark_theme.qss
│       ├── data/                    # Auto-created directories
│       ├── version.json
│       └── ... (PySide6 DLLs, Python runtime)
```

### Lưu ý đóng gói

- **ML models bị exclude** trong spec file (torch, transformers, whisper) để giảm kích thước build (~200MB thay vì 2GB+). Chúng được load at runtime nếu user đã cài pip.
- Nếu cần **standalone build** (bao gồm mọi thứ), bỏ phần `excludes` trong `emotion_studio.spec`.

---

## 6. Tạo Windows Installer (Inno Setup)

### Yêu cầu

- [Inno Setup 6](https://jrsoftware.org/isdl.php) — cài vào `C:\Program Files (x86)\Inno Setup 6\`

### Tạo installer

**Cách 1: Qua build script** (đã bao gồm)
```powershell
.\build\build.ps1 -Version "1.0.0"    # Tự động gọi Inno Setup ở step 5
```

**Cách 2: Thủ công**
```bash
# Mở Inno Setup → Compile
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DMyAppVersion=1.0.0 installer\emotion_data_studio.iss
```

### Kết quả

```
installer/output/
└── EmotionDataStudio-1.0.0-Setup.exe    # ★ Windows installer
```

### Installer thực hiện gì

1. Copy app bundle vào `%ProgramFiles%\Emotion Data Studio\`
2. Tạo thư mục data với quyền write
3. Đặt biến môi trường: `EDS_DATA_DIR`, `EDS_FFMPEG_PATH`
4. Tạo shortcut Desktop + Start Menu
5. Đăng ký registry (version, install path)

---

## 7. Deploy Cloud Run (Google Cloud)

### 7.1 Yêu cầu

- [Google Cloud account](https://console.cloud.google.com/) với credit hoạt động
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) đã cài và auth

### 7.2 Setup ban đầu (chỉ 1 lần)

```bash
# Login
gcloud auth login

# Tạo project (hoặc chọn existing)
gcloud projects create eds-project --name="Emotion Data Studio"
gcloud config set project eds-project

# Enable APIs
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    sqladmin.googleapis.com \
    storage.googleapis.com

# Tạo Cloud SQL instance (PostgreSQL)
gcloud sql instances create eds-postgres \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=asia-southeast1 \
    --storage-size=10GB

# Tạo database
gcloud sql databases create emotion_studio --instance=eds-postgres

# Tạo user
gcloud sql users create eds_user \
    --instance=eds-postgres \
    --password=YOUR_SECURE_PASSWORD

# Tạo GCS bucket
gcloud storage buckets create gs://eds-data-bucket \
    --location=asia-southeast1 \
    --uniform-bucket-level-access

# Tạo service account cho desktop app
gcloud iam service-accounts create eds-desktop \
    --display-name="EDS Desktop Client"

# Cấp quyền
gcloud projects add-iam-policy-binding eds-project \
    --member="serviceAccount:eds-desktop@eds-project.iam.gserviceaccount.com" \
    --role="roles/cloudsql.client"

gcloud projects add-iam-policy-binding eds-project \
    --member="serviceAccount:eds-desktop@eds-project.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# Tải key file cho desktop app
gcloud iam service-accounts keys create service-account.json \
    --iam-account=eds-desktop@eds-project.iam.gserviceaccount.com
```

### 7.3 Deploy lên Cloud Run

**Cách 1: Dùng Cloud Build (CI/CD — khuyến nghị)**

```bash
cd tools/emotion-data-studio

# Deploy
gcloud builds submit \
    --config deploy/cloudbuild.yaml \
    --substitutions=_CLOUD_SQL_INSTANCE="eds-project:asia-southeast1:eds-postgres"
```

**Cách 2: Deploy trực tiếp**

```bash
cd tools/emotion-data-studio

# Build Docker image
docker build -t eds-api -f deploy/Dockerfile .

# Tag
docker tag eds-api gcr.io/eds-project/eds-api:latest

# Push
docker push gcr.io/eds-project/eds-api:latest

# Deploy
gcloud run deploy eds-api \
    --image gcr.io/eds-project/eds-api:latest \
    --region asia-southeast1 \
    --memory 2Gi \
    --cpu 2 \
    --max-instances 3 \
    --allow-unauthenticated \
    --add-cloudsql-instances eds-project:asia-southeast1:eds-postgres \
    --set-env-vars "ENV=production,EDS_DATA_DIR=/app/data"
```

### 7.4 Kiểm tra deployment

```bash
# Lấy URL
gcloud run services describe eds-api --region=asia-southeast1 --format="value(status.url)"

# Test health
curl https://your-service-url.run.app/health
# Expected: {"status":"healthy","version":"1.0.0"}

# Test status
curl https://your-service-url.run.app/api/status
```

### 7.5 Ước tính chi phí (Google Cloud $300 credit)

| Dịch vụ | Cấu hình | Chi phí ước tính/tháng |
|---|---|---|
| Cloud Run | 2 CPU, 2GB RAM, 0-3 instances | ~$5-15 (pay per use) |
| Cloud SQL | db-f1-micro, 10GB | ~$10-15 |
| Cloud Storage | 50GB data | ~$1-2 |
| **Tổng** | | **~$16-32/tháng** |

> 💡 Với $300 credit, bạn có thể chạy **khoảng 10-18 tháng** ở mức sử dụng nhẹ.

---

## 8. Setup Auto-Updater (Cloudflare R2)

### 8.1 Tạo R2 bucket

1. Đăng nhập [Cloudflare Dashboard](https://dash.cloudflare.com/)
2. Vào R2 Object Storage → Create Bucket → tên: `eds-releases`
3. Vào Settings → Public Access → Enable (hoặc setup custom domain)

### 8.2 Upload release

Tạo file `latest.json`:

```json
{
    "version": "1.1.0",
    "download_url": "https://releases.your-domain.com/EmotionDataStudio-1.1.0-Setup.exe",
    "release_notes": "- Cải thiện hiệu năng AI pipeline\n- Thêm keyboard shortcuts\n- Fix crash khi export",
    "file_size": 85000000,
    "sha256": "abc123..."
}
```

Upload lên R2:
```bash
# Cài wrangler (Cloudflare CLI)
npm install -g wrangler

# Login
wrangler login

# Upload
wrangler r2 object put eds-releases/latest.json --file=latest.json
wrangler r2 object put eds-releases/EmotionDataStudio-1.1.0-Setup.exe --file=path/to/installer.exe
```

### 8.3 Cấu hình trong app

Đặt URL vào `.env`:
```env
EDS_UPDATE_URL=https://releases.your-domain.com
```

App sẽ tự động kiểm tra `{EDS_UPDATE_URL}/latest.json` mỗi lần khởi động.

---

## 9. Xử lý lỗi thường gặp

### Lỗi import PySide6

```
ImportError: DLL load failed while importing QtWidgets
```

**Giải pháp**: Cài đặt [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Lỗi FFmpeg không tìm thấy

```
FileNotFoundError: ffmpeg not found
```

**Giải pháp**: Thêm FFmpeg vào PATH hoặc đặt trong `.env`:
```env
EDS_FFMPEG_PATH=C:\ffmpeg\bin\ffmpeg.exe
```

### Lỗi CUDA out of memory

```
torch.cuda.OutOfMemoryError
```

**Giải pháp**: 
- Giảm batch size trong `backend/config.py`
- Hoặc tắt GPU: đặt `USE_GPU=false` trong `.env`

### Lỗi database locked

```
sqlite3.OperationalError: database is locked
```

**Giải pháp**: Đảm bảo chỉ **1 instance** của app đang chạy. SQLite dùng `check_same_thread=False` cho QThread nhưng không hỗ trợ multi-process.

### Lỗi Cloud SQL connection

```
Could not connect to Cloud SQL
```

**Giải pháp**:
1. Kiểm tra Cloud SQL Auth Proxy đang chạy: `cloud-sql-proxy eds-project:asia-southeast1:eds-postgres`
2. Kiểm tra credentials: `echo $GOOGLE_APPLICATION_CREDENTIALS`
3. Kiểm tra firewall/VPC settings

---

## Quick Reference

```bash
# === Development ===
cd tools/emotion-data-studio
python app.py                          # Chạy desktop app

# === Build ===
.\build\build.ps1 -Version "1.0.0"    # Build + installer

# === Cloud ===
gcloud builds submit --config deploy/cloudbuild.yaml   # Deploy Cloud Run
curl https://your-url.run.app/health                    # Health check

# === Test ===
python -c "from ui.main_window import MainWindow; print('OK')"   # Test imports
python -m pytest tests/                                            # Run tests
```
