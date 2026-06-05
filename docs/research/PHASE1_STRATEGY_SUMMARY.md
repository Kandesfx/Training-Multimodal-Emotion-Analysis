# Chiến Lược Phase 1 — Multimodal Emotion Analysis

## Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức

> **Trạng thái:** Đang thảo luận kiến trúc — CHƯA final
> **Ngày tạo:** 2026-06-05
> **Ghi chú:** Document này tổng hợp toàn bộ bối cảnh, tài nguyên, và đề xuất. Cần thống nhất kiến trúc trước khi triển khai.

---

## 1. Bối Cảnh & Yêu Cầu

### 1.1 Yêu Cầu Đề Tài (từ `yeu_cau_de_tai.md`)

| Tiêu chí | Yêu cầu |
|:---|:---|
| **Thuật toán bắt buộc** | Các thuật toán Deep Learning đã học: CNN, RNN, LSTM |
| **Thuật toán gợi ý (cộng điểm)** | CNN, RNN, LSTM |
| **Code** | Đầy đủ, rõ ràng, tường minh, có quá trình training |
| **Ứng dụng** | Giao diện + vận hành được thực tế (4.0 điểm) |
| **Báo cáo** | Đầy đủ theo quy định (2.0 điểm) |

### 1.2 Yêu Cầu Kỹ Thuật Từ Người Dùng

- **Kiểm soát hoàn toàn**: Toàn bộ quá trình, thông số, lịch sử train phải rõ ràng
- **Tường minh**: Công nghệ, phương pháp tích hợp phải được ghi chép cụ thể (phục vụ báo cáo)
- **Tài nguyên máy**:
  - Local: GPU yếu, CPU đa nhân/đa luồng
  - Máy dự phòng ở xa: RTX 5070
  - Google Cloud: có tài nguyên mạnh
- **$300 GCP credits**: Dùng cho Colab Pro + GPU
- **Thứ tự**: Phase 1 trước, cải thiện EDS sau
- **Thuật toán**: Dùng đề xuất nhưng có thể dùng biến thể mạnh hơn

---

## 2. Hiện Trạng Dữ Liệu

### 2.1 CMU-MOSEI Dataset (Đã có trên đĩa)

```
data/MSA-Dataset/
├── aligned_50.pkl        → 4.6 GB  (word-aligned, 22,856 mẫu)
│   ├── train: 16,326 mẫu
│   ├── valid: 1,871 mẫu
│   └── test:  4,659 mẫu
│   └── Keys: text(50,768), audio(50,74), vision(50,35), regression_labels
├── unaligned_50.pkl      → 13.6 GB (unaligned)
├── Raw.zip               → 27.3 GB (video clips gốc)
└── Git/MMSA/            → MMSA framework (20+ model architectures)
```

### 2.2 Feature Dimensions

| Modality | Key trong .pkl | Shape | Công cụ trích xuất | Chiều |
|:---|:---|:---|:---|:---|
| Text | `'text'` | `(N, 50, 768)` | BERT-base-uncased | 768 |
| Audio | `'audio'` | `(N, 50, 74)` | COVAREP | 74 |
| Vision | `'vision'` | `(N, 50, 35)` | FACET (35 AUs) | 35 |
| Label | `'regression_labels'` | `(N,)` float | Human annotation | [-3.0, +3.0] |

---

## 3. Mục Tiêu Phase 1

### 3.1 Mục tiêu kỹ thuật

| Giai đoạn | Mục tiêu | Metrics |
|:---|:---|:---|
| Baseline | Verify pipeline hoạt động | Loss giảm, gradient flow đúng |
| Early Fusion | Baseline có thể so sánh | MAE < 0.65 |
| MulT | Mô hình chính | MAE < 0.58, Corr > 0.72 |
| Nâng cao | Biến thể mạnh hơn | MAE < 0.55 |

### 3.2 Mục tiêu báo cáo

- Trình bày rõ: Input → Feature Extraction → Model Architecture → Training → Evaluation
- Ghi chép đầy đủ hyperparameters, training logs
- So sánh nhiều models (ablation study)
- Visualization: confusion matrix, training curves, attention weights

---

## 4. Đề Xuất Kiến Trúc (CHƯA FINAL — CẦN THẢO LUẬN)

### 4.1 Đề xuất: MulT-Based Multimodal Fusion

**Lý do chọn MulT:**
1. Sử dụng trực tiếp pre-extracted features (768/74/35 chiều) — không cần fine-tune BERT, tiết kiệm VRAM
2. Cross-modal Transformer attention dễ trình bày trong báo cáo (CNN, RNN, LSTM đều xuất hiện)
3. Baseline reference từ MMSA: MulT đạt MAE=55.93, Corr=73.31 trên MOSEI
4. Cross-modal attention cho phép model tự học modality nào quan trọng hơn

### 4.2 Sơ đồ kiến trúc đề xuất

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Pre-extracted Features (aligned_50.pkl)                 │
│  Text: (B, 50, 768)     — BERT embeddings                     │
│  Audio: (B, 50, 74)    — COVAREP features                   │
│  Vision: (B, 50, 35)   — FACET Action Units                 │
└───────────────────────────┬─────────────────────────────────────┘
                            │
           ┌────────────────┼────────────────┐
           ▼                ▼                ▼
     ┌───────────┐   ┌───────────┐   ┌───────────┐
     │  Text     │   │  Audio    │   │  Vision   │
     │  Linear   │   │  Linear   │   │  Linear   │
     │  768→256  │   │  74→256   │   │  35→256   │
     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
           │                │                │
           ▼                ▼                ▼
     ┌───────────┐   ┌───────────┐   ┌───────────┐
     │  Pos Emb  │   │  Pos Emb  │   │  Pos Emb  │
     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
           │                │                │
           └────────┬───────┘────────┬───────┘
                    │               │
                    │  Cross-Modal │
                    │  Attention    │
                    │  (MulT)       │
                    │               │
                    ▼               ▼
            ┌──────────────────────────────────┐
            │   Feature Fusion: Concat (256×3) │
            │   → Linear(768 → 256)           │
            │   → LayerNorm + Dropout         │
            │   → Linear(256 → 64)            │
            │   → LayerNorm + Dropout         │
            └────────────────┬───────────────┘
                              ▼
            ┌──────────────────────────────────┐
            │   Output Layer                    │
            │   → Linear(64 → 1)              │
            │   → Regression ∈ [-3, +3]         │
            └──────────────────────────────────┘
```

### 4.3 Cross-Modal Attention Detail

```
Text (Q) ──────────► Cross-Attn ◄── Audio (K, V) ──► Audio→Text
Text (Q) ──────────► Cross-Attn ◄── Vision (K, V) ──► Vision→Text
```

---

## 5. Đề Xuất Các Models Để Implement

### 5.1 Models theo thứ tự ưu tiên

| # | Model | Độ phức tạp | Mô tả | Target |
|:---|:---|:---|:---|:---|
| 1 | **Early Fusion LSTM** | Thấp | 3 BiLSTMs → Concat → FC | Baseline verify |
| 2 | **MulT** | Trung bình | Cross-modal Transformer | MAE < 0.58 |
| 3 | **SELF_MM variant** | Cao | Self-supervised multi-task | MAE < 0.55 |
| 4 | **Ensemble** | — | Kết hợp tất cả | Tốt nhất |

### 5.2 Baseline reference từ MMSA (CMU-MOSEI)

| Model | Has0_acc_2 | MAE | Corr | Data Setting |
|:---|:---|:---|:---|:---|
| ef_lstm | 77.84 | 60.05 | 68.25 | Aligned |
| lf_dnn | 80.60 | 58.02 | 70.87 | Unaligned |
| lmf | 80.54 | 57.57 | 71.69 | Unaligned |
| mfn | 78.94 | 57.33 | 71.82 | Aligned |
| mult | 81.15 | 55.93 | 73.31 | Unaligned |
| self_mm | 83.76 | 53.09 | 76.49 | Unaligned |
| tetfn | 84.12 | 53.73 | 76.96 | Aligned |
| cenet | 83.52 | 52.59 | 77.75 | Unaligned |

> **Lưu ý:** Giá trị MAE trong MMSA được nhân 100 (vd MAE=60.05 nghĩa là 0.6005)

---

## 6. Lộ Trình Thực Hiện Đề Xuất

```
TUẦN 1: Setup + Baseline chạy được
  ├── Mount Google Drive vào Colab
  ├── Viết dataset loader (đọc .pkl)
  ├── Viết 3 unimodal encoders (BiLSTM)
  ├── Baseline: Early Fusion = Concat → FC → Regression
  ├── Train 10 epochs → verify gradient flow
  └── Target: Đạt MAE < 0.70

TUẦN 2: Nâng cấp lên MulT
  ├── Thêm Positional Encoding + Linear Projection
  ├── Implement Cross-Modal Transformer Encoder
  ├── Train 50 epochs với EarlyStopping (patience=8)
  └── Target: Đạt MAE < 0.60, Corr > 0.72

TUẦN 3: Tinh chỉnh & Benchmark
  ├── Hyperparameter tuning (lr, dropout, batch size)
  ├── Ablation study: V-only, A-only, T-only, V+A, V+T, A+T
  ├── Thêm attention visualization
  └── Target: MAE ~0.55-0.58, Corr ~0.73-0.76

TUẦN 4: Đánh giá & Báo cáo
  ├── Đánh giá cuối cùng trên test set
  ├── So sánh với MMSA baseline
  └── Viết báo cáo kỹ thuật
```

---

## 7. Tài Nguyên Tính Toán

### 7.1 Chiến lược sử dụng

| Nền tảng | Sử dụng cho | Chi phí |
|:---|:---|:---|
| **Colab (T4 free)** | Baseline testing, debug | Miễn phí |
| **Colab Pro (A100)** | Training chính (50 epochs) | GCP credits |
| **Local (CPU)** | Không khuyến khích (quá chậm) | — |

### 7.2 Ước tính thời gian train (Colab T4)

| Model | Epochs | Thời gian |
|:---|:---|:---|
| Early Fusion (BiLSTM) | 10 | ~20-30 phút |
| MulT | 50 | ~2-3 giờ |
| Ablation + Tuning | — | ~2 giờ |

---

## 8. Cấu Trúc Code Đề Xuất

```
training/
├── config_phase1.py          # Cấu hình Phase 1 (HPO, paths, hyperparameters)
├── dataset_mosei.py          # PyTorch Dataset (đọc .pkl, chia train/valid/test)
├── models/
│   ├── __init__.py
│   ├── unimodal_encoder.py   # BiLSTM cho text/audio/vision
│   ├── early_fusion.py       # Baseline: Concat → FC
│   ├── mult.py              # MulT: Cross-modal Transformer
│   └── self_mm_variant.py   # Self-supervised multi-task (nâng cao)
├── trainer.py                # Training loop + logging + checkpointing
├── evaluator.py             # Metrics: MAE, Corr, Acc-2/5/7, F1
└── main.py                  # Entry point

notebooks/
├── 01_dataset_exploration.ipynb    # Khám phá dữ liệu .pkl
├── 02_baseline_early_fusion.ipynb # Baseline model
├── 03_mult_training.ipynb          # MulT training
└── 04_evaluation.ipynb            # Đánh giá & visualization

logs/                         # TensorBoard + CSV logs
checkpoints/                  # Model checkpoints
```

---

## 9. Metrics Chuẩn (Theo MMSA)

| Metric | Mô tả | Công thức |
|:---|:---|:---|
| **MAE** | Mean Absolute Error | $\frac{1}{N}\sum|y - \hat{y}|$ |
| **Corr** | Pearson Correlation | $\frac{\sum(y-\bar{y})(\hat{y}-\bar{\hat{y}})}{\sqrt{\sum(y-\bar{y})^2}\sqrt{\sum(\hat{y}-\bar{\hat{y}})^2}}$ |
| **Acc-2 (Has0)** | Nhị phân (≥0 vs <0) | Accuracy trên Positive/Negative |
| **Acc-5** | 5 lớp [-2..+2] | Làm tròn regression → 5 lớp |
| **Acc-7** | 7 lớp [-3..+3] | Làm tròn regression → 7 lớp |
| **F1** | F1-score nhị phân | Weighted F1 trên Positive/Negative |

---

## 10. Open Questions (Cần Thảo Luận)

1. **Kiến trúc cuối cùng**: MulT hay biến thể khác? (EF_LSTM → MulT → ?)
2. **Phạm vi báo cáo**: Chỉ Phase 1 hay đề cập cả Phase 2 trong đồ án?
3. **Dataset cho Phase 2**: Có cần thiết kế ngay hay chờ Phase 1 xong?
4. **Baseline comparison**: Có cần so sánh với MMSA framework có sẵn không?
5. **Feature extraction**: Dùng aligned hay unaligned? (Aligned đã có word-level alignment)

---

## 11. Tài Liệu Liên Quan

- `docs/research/TRAINING_ROADMAP.md` — Lộ trình 2 giai đoạn
- `docs/research/FINE_TUNING_STRATEGY.md` — Chiến lược Transfer Learning
- `docs/research/DATA_COLLECTION_STRATEGY.md` — Thu thập dữ liệu tiếng Việt
- `docs/research/DATASET_PREPARATION.md` — Chẩn đoán CMU-MOSEI
- `docs/research/phan_tich_training_da_phuong_thuc.md` — Phân tích phương pháp SOTA
- `docs/architecture/ARCHITECTURE.md` — Kiến trúc tổng thể hệ thống
- `data/MSA-Dataset/Git/MMSA/` — MMSA Framework (baseline reference)
- `training/config.py` — Config dataclasses hiện tại
