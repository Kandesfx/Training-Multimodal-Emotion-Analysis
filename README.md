# 🎭 BCDA — Multimodal Emotion Recognition

> Đồ án nhận diện cảm xúc đa phương thức (video + audio + text) cho tiếng Việt.

## 📁 Cấu trúc dự án

```
BCDA/
├── training/                  # Code train model
├── models/                    # Kiến trúc model (định nghĩa mạng)
├── notebooks/                 # Jupyter notebooks (thử nghiệm)
├── checkpoints/               # Model weights (gitignored)
├── tests/                     # Tests
├── docs/                      # Tài liệu
│   ├── architecture/          # Tài liệu kiến trúc
│   └── SETUP_GUIDE.md         # Hướng dẫn cài đặt
│
└── tools/
    └── emotion-data-studio/   # Tool chuẩn bị dataset
        ├── README.md           # Hướng dẫn riêng tool
        └── ...
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/your-team/BCDA.git
cd BCDA

# Tạo venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt dependencies (model training)
pip install -r requirements.txt

# Cài đặt tool EDS (chuẩn bị data)
cd tools/emotion-data-studio
pip install -r requirements.txt
python app.py
```

## 📚 Tài liệu

- [Hướng dẫn Setup & Deploy](docs/SETUP_GUIDE.md)
- [Kiến trúc Emotion Data Studio](docs/architecture/EMOTION_DATA_STUDIO.md)
- [Kiến trúc Model](docs/architecture/ARCHITECTURE.md)

## 👥 Team

BCDA Team — Deep Learning Course Project
