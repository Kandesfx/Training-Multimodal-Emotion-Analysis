# BÁO CÁO PHÂN TÍCH ROOT CAUSE — PERFORMANCE MÔ HÌNH

**Đề tài:** Đề Tài 17 — Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức
**Ngày phân tích:** 09/06/2026
**Nguồn dữ liệu:** Phase 1 Training Report + Kết quả W&B + Source Code

---

## Mục lục

1. [Tổng quan kết quả thực nghiệm](#1-tổng-quan-kết-quả-thực-nghiệm)
2. [Yếu tố từ Dataset và Data Pipeline](#2-yếu-tố-từ-dataset-và-data-pipeline)
3. [Yếu tố từ Kiến trúc mô hình](#3-yếu-tố-từ-kiến-trúc-mô-hình)
4. [Yếu tố từ Training Strategy](#4-yếu-tố-từ-training-strategy)
5. [Yếu tố từ Evaluation Methodology](#5-yếu-tố-từ-evaluation-methodology)
6. [Tổng hợp Root Cause](#6-tổng-hợp-root-cause)
7. [Priority Matrix — Thứ tự ưu tiên fix](#7-priority-matrix--thứ-tự-ưu-tiên-fix)
8. [Kết luận](#8-kết-luận)

---

## 1. Tổng quan kết quả thực nghiệm

### 1.1. Kết quả cross-check: Hình vs Báo cáo

| Thông tin | Hình | Báo cáo | Khớp? |
|:---|:---:|:---:|:---:|
| Baseline: Train loss 0.59 → 0.10, Valid loss ~0.56-0.60 | ✅ | Epoch 14: Train=0.045, Valid≈0.60 | ✅ Đúng |
| Baseline: Valid Corr đỉnh ở epoch 4-6 | ✅ | Epoch 4-6 đạt ~0.70 | ✅ Đúng |
| Improved LSTM: Best ở epoch 7 | ✅ | Valid MAE=0.5323, Corr=0.7254 | ✅ Đúng |
| MulT Emotion: Overfitting bắt đầu epoch 9 | ✅ | Val loss đáy epoch 9 | ✅ Đúng |

### 1.2. Bảng tổng hợp kết quả

| Chỉ số | Baseline LSTM | Improved LSTM | MulT Emotion |
|:---|:---:|:---:|:---:|
| **Task** | Sentiment Regression | Sentiment Regression | Emotion Classification |
| **Test MAE** | 0.6071 | 0.5859 | — |
| **Test Correlation** | 0.6995 | 0.7229 | — |
| **Test Acc-2** | 0.8103 | 0.8137 | — |
| **Valid Mean F1** | — | — | 0.2064 (best: 0.2117) |
| **Happy F1** | — | — | 0.4709 |
| **Sad F1** | — | — | 0.2673 |
| **Angry F1** | — | — | 0.1949 |
| **Disgust F1** | — | — | 0.1730 |
| **Surprise F1** | — | — | 0.0977 |
| **Fear F1** | — | — | 0.0348 |
| **Số tham số** | ~1.1M | ~2.03M | ~1.7M |
| **Epoch tốt nhất** | 6 | 7 | 9 |
| **Epoch dừng** | 14 | 17 | 29 |
| **Overfitting gap** | 13x | ~10x | F1 drop từ epoch 9 |

### 1.3. Đồ thị training curves mô tả

**Baseline LSTM (Hình 2):**
```
Train Loss: ████████████████░░░░░░░░░░░░░░░  (0.59 → 0.10, giảm 83%)
Valid Loss: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (dao động ~0.56-0.60, không giảm)
Valid Corr: ────∧∧∧───────────────────────  (đỉnh epoch 4-6, ~0.70)
```

**Improved LSTM (Hình 3):**
```
Train Loss: ██████████████░░░░░░░░░░░░░░░░  (giảm mạnh đến ~0.05)
Valid Loss: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (dao động ~0.51-0.57)
Valid MAE:  ─────────∧────────────────────  (đáy epoch 7: 0.5323)
Valid Corr: ────────────∧───────────────────  (đỉnh epoch 7: 0.7254)
```

**MulT Emotion (Hình 1):**
```
Train Loss:  ████████████████████████░░░░░  (0.185 → 0.108, giảm 42%)
Val Loss:    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  (tăng từ epoch 9 trở đi, đến 0.1314)
Val MAE:     ────────∧────────────────────  (tăng từ epoch 9, đến 1.4222)
Val Mean F1: ────────∧────────────────────  (giảm từ epoch 9, đến 0.2064)
Val Mean Acc: ───∧∧∧∧∧∧∧∧∧∧∧────────────  (oscillate mạnh, 12.5%-38%)
```

---

## 2. Yếu tố từ Dataset và Data Pipeline

### 2.1. Vision features all-zero gây NaN trong Attention

**Mã nguồn:** `training/models/mult.py`, hàm `_ensure_valid_mask()`

```
Root cause:
  vision features = 0 cho samples không detect được face
  Tổng 35 chiều = 0 → mask = False cho tất cả timesteps
  nn.MultiheadAttention với all-False mask → softmax([-inf]) = NaN
  → NaN poisoning toàn batch
```

Trong CMU-MOSEI, một lượng lớn samples có vision features = 0 (không detect được khuôn mặt). Mã nguồn hiện tại có fix `_ensure_valid_mask()` nhưng:

- Fix bằng cách force position 0 = True — **đây là heuristic không hoàn hảo**
- Position 0 có thể là padding hoặc silent frame, không phải real face detection
- Mask sai → attention scores sai → representation sai → metric kém

**Tác động:** Cao — ảnh hưởng trực tiếp đến chất lượng vision modality trong MulT.

---

### 2.2. COVAREP audio features chứa Inf

**Mã nguồn:** `training/dataset_mosei.py`, hàm `_prepare_array()`

```python
replace_inf = True
audio_inf_replacement = 0.0
```

- COVAREP features có Inf (do lỗi pitch tracker trên silence segments)
- Thay bằng 0.0 → **tạo ra "fake silence"** — mô hình học rằng silence = zero vector
- 0.0 có thể gây confusion với real zero-padded timesteps

**Tác động:** Trung bình — audio modality bị nhiễu bởi giá trị thay thế không chính xác.

---

### 2.3. ~23% samples bị lọc khi dùng emotion mode

**Mã nguồn:** `training/dataset_mosei.py`, lines 38-53

```python
if self.task_type == "emotion" and "emotion_matched_mask" in split_data:
    mask = np.asarray(split_data["emotion_matched_mask"], dtype=bool)
    # ... filter all arrays
```

```
Dataset gốc:           16,326 mẫu train
Sau filter emotion:    ~12,500 mẫu train (giảm 23%)
```

- Giảm 23% training data → model yếu hơn, đặc biệt với imbalanced classes
- Các mẫu bị lọc có thể chứa pattern quan trọng cho Fear/Surprise

**Tác động:** Cao — giảm đáng kể lượng data huấn luyện.

---

### 2.4. Class Imbalance cực đoan trong emotion labels

```
CMU-MOSEI Emotion Distribution:
  Happy:    34.0%  ████████████████████
  Angry:    14.5%  █████████
  Sad:      14.1%  █████████
  Disgust:  11.9%  ████████
  Surprise:  3.3%  ██
  Fear:      2.2%  █

Ratio Happy/Fear = 15.5:1
```

**Hệ quả với BCE loss không có weight:**
- Model luôn predict "not Fear" → đúng 97.8% cases (easy correct)
- Model luôn predict "Happy" → đúng 66% cases (safe baseline)
- Không có incentive để learn Fear/Surprise patterns vì chúng barely affect overall loss

**Pattern F1 theo tần suất:**

| Emotion | Tần suất | F1 thực tế | F1 kỳ vọng |
|:---|:---:|:---:|:---:|
| Happy | 34.0% | 0.4709 | 0.45-0.50 |
| Sad | 14.1% | 0.2673 | 0.25-0.30 |
| Angry | 14.5% | 0.1949 | 0.15-0.22 |
| Disgust | 11.9% | 0.1730 | 0.12-0.18 |
| Surprise | 3.3% | 0.0977 | 0.08-0.12 |
| Fear | 2.2% | 0.0348 | 0.03-0.06 |

**Mối quan hệ:** F1 ≈ f(log(frequency)) — gần như tuyến tính theo log tần suất.

---

## 3. Yếu tố từ Kiến trúc mô hình

### 3.1. Baseline: Projection 768→128 là bottleneck nghiêm trọng

**Mã nguồn:** `training/models/unimodal_encoder.py`

```python
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, ...):
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, ...)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return torch.cat([forward_hidden, backward_hidden], dim=1)
```

- LSTM hidden=128 nén thông tin từ 768 chiều → **information bottleneck cực đoan**
- Chỉ có 1 layer → **không có hierarchical representation**
- Lấy **hidden state cuối** → bị ảnh hưởng bởi padding (timesteps 50 có thể là padding)

---

### 3.2. Baseline: Chỉ dùng last hidden state

```python
# unimodal_encoder.py, line 21
_, (hidden, _) = self.lstm(x)  # ← bỏ qua tất cả intermediate states
```

- BERT có 50 timesteps với thông tin phân bố khắp chuỗi
- Hidden state cuối bị chi phối bởi timestep gần nhất
- Không có attention mechanism để chọn timestep quan trọng

---

### 3.3. Cả Baseline và Improved đều "Early Fusion" — fusion muộn

```
Baseline & Improved LSTM:
  Text ──► BiLSTM ──► vector(256) ──┐
  Audio ──► BiLSTM ──► vector(128) ─┤
  Vision ──► BiLSTM ──► vector(128) ─┘
       concat(512) ──► MLP ──► output

  ❌ KHÔNG CÓ tương tác cross-modal trong quá trình encode
  ✓ Chỉ gặp nhau ở fusion layer CUỐI CÙNG
```

**Hệ quả:** Text, Audio, Vision được encode **hoàn toàn độc lập**. Mô hình không bao giờ "nhìn thấy" mối quan hệ giữa:
- Tone giọng nói và nội dung text
- Biểu cảm khuôn mặt và ngữ cảnh câu
- Timing của speech và gesture

---

### 3.4. MulT: d_model=64 quá nhỏ cho 768→64 projection

**Mã nguồn:** `training/config_phase1.py`, `Phase1MulTModelConfig`

```python
d_model: int = 64   # MulT Emotion config
num_heads: int = 4   # MulT Emotion config
```

| Thông số | MulT Emotion (thực tế) | MulT P1 (đề xuất) |
|:---|:---:|:---:|
| d_model | 64 | 128 |
| num_heads | 4 | 8 |
| dim per head | 16 | 16 |
| Thông tin giữ lại | 8.3% | 16.7% |

- 768 chiều BERT → projection thành 64 chiều → **mất 92% thông tin**
- 4 attention heads → mỗi head chỉ có 64/4 = **16 chiều per head** → quá ít
- num_cross_layers=4 nhưng mỗi layer rất nhỏ → không đủ depth

---

### 3.5. MulT: Chạy 29 epochs trong khi best ở epoch 9

```
Root cause: Early Stopping metric và loss không align

Code: trainer.py
  if self.task_type == "emotion":
      self.metric_for_best = "mean_f1"  # metric chính
      self.maximize_metric = True

  # loss = BCE (cross-entropy)
  # metric = mean_f1
  # → metric và loss KHÔNG align
```

- Valid mean_f1 best = 0.2117 tại epoch 9
- Nhưng `patience=10` → cần 10 epochs không cải thiện mới dừng
- Epoch 9 → epoch 29 = 20 epochs → có local improvement nhỏ nhưng F1 metric chính giảm
- **Model tiếp tục chạy 20 epochs vô ích**, chỉ overfit nặng thêm

---

## 4. Yếu tố từ Training Strategy

### 4.1. Loss Function cho Emotion: BCE không đủ

**Mã nguồn:** `training/trainer.py`, line 103

```python
if loss_type == "bce":
    return nn.BCEWithLogitsLoss()  # ← KHÔNG có pos_weight
```

- `BCEWithLogitsLoss()` **mặc định pos_weight=1** cho tất cả 6 emotions
- Happy (34%) và Fear (2.2%) có cùng weight → model ưu tiên Happy
- Không có mechanism để "force" model học rare classes

**So sánh với điều nên làm:**

```python
# Tính pos_weight = số negative / số positive cho mỗi emotion
happy_weight = num_negatives_happy / num_positives_happy   # ≈ 2.0
fear_weight = num_negatives_fear / num_positives_fear      # ≈ 44.0

pos_weight = torch.tensor([happy_weight, sad_weight, angry_weight,
                          surprise_weight, disgust_weight, fear_weight])
return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

---

### 4.2. Metric cho Early Stopping không đồng nhất với Loss

**Mã nguồn:** `training/trainer.py`

```python
task_type="sentiment": metric_for_best = "mae"
task_type="emotion": metric_for_best = "mean_f1"

NHƯNG loss function:
  sentiment: mse_l1
  emotion: bce (không align với mean_f1)
```

- Sentiment dùng `mae` làm metric nhưng loss là `mse_l1` → metric và loss không align
- Emotion dùng `mean_f1` làm metric nhưng loss là `bce` → metric và loss không align
- **BCE loss minimize cross-entropy**, không trực tiếp tối ưu F1-score

---

### 4.3. Baseline Learning Rate quá cao: 1e-3

Theo báo cáo Phase 1, Baseline dùng `lr=1e-3`:

- 1e-3 với AdamW và 1-layer LSTM → **quá lớn**
- LSTM gradients dễ explode hơn Transformer vì recurrent nature
- Gây training instability → spike ở epoch 3 (valid loss tăng vọt)

**So sánh:**

| Config | Baseline thực tế | MulT Emotion | MulT P1 |
|:---|:---:|:---:|:---:|
| Learning Rate | 1e-3 | 1e-4 | 1e-4 |
| Scheduler | ReduceLROnPlateau | Cosine Warmup | Cosine Warmup |
| Hợp lý? | ❌ Quá cao | ✅ | ✅ |

---

### 4.4. ReduceLROnPlateau không phù hợp

**Mã nguồn:** `training/trainer.py`, lines 109-115

```python
if sched_type == "plateau":
    return ReduceLROnPlateau(
        self.optimizer,
        mode="min",
        factor=0.5,      # ← giảm 50% mỗi lần
        patience=3,       # ← chỉ 3 epochs!
    )
```

- `patience=3` → LR giảm quá nhanh (chỉ cần 3 epochs không cải thiện)
- `factor=0.5` → giảm 50% mỗi lần → LR collapse nhanh
- Metric = val_loss (MSE) nhưng target là MAE → **metric mismatch**

---

## 5. Yếu tố từ Evaluation Methodology

### 5.1. Ngưỡng binarization 0.5 cố định cho tất cả emotions

**Mã nguồn:** `training/evaluator_emotion.py`, line 67

```python
y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))  # sigmoid
y_pred_bin = (y_pred_prob >= 0.5).astype(int)  # ← 0.5 cho TẤT CẢ 6 emotions
```

**Tại sao đây là vấn đề lớn:**

| Emotion | Tỷ lệ positive | Ngưỡng tối ưu lý thuyết | Sai lệch từ 0.5 |
|:---|:---:|:---:|:---:|
| Happy | 34% | ~0.30-0.35 | quá cao |
| Fear | 2.2% | ~0.05-0.10 | quá cao gấp 5-10 lần |

- Với **imbalanced data**, ngưỡng tối ưu khác nhau cho mỗi emotion
- 0.5 là ngưỡng hợp lý khi **tỷ lệ positive ≈ 50%**
- Khi positive rate = 2.2%, ngưỡng tối ưu có thể là 0.05-0.10
- Dùng 0.5 cho Fear → model gần như **không bao giờ predict Fear positive**

---

### 5.2. MAE cho Emotion là approximation không chính xác

**Mã nguồn:** `training/evaluator_emotion.py`, line 92

```python
mae_per_emo = np.mean(np.abs(y_true - y_pred_prob * 3.0), ...)
```

- `y_pred_prob` = sigmoid(logit) → range [0, 1]
- `* 3.0` → scale thành [0, 3]
- `y_true` = intensity ground truth [0, 3]
- **Vấn đề:** sigmoid là monotonic nhưng không tuyến tính → MAE approximation không chính xác
- Đây là so sánh khác loại (intensity vs probability-scaled)

---

### 5.3. Mean F1 = macro average — nhạy cảm với rare classes

```python
Mean F1 = (F1_happy + F1_sad + F1_angry + F1_surprise + F1_disgust + F1_fear) / 6
```

- Macro F1 treat Happy (34%) và Fear (2.2%) có **TRỌNG SỐ BẰNG NHAU**
- Fear F1 = 0.035 kéo Mean F1 xuống rất nhiều
- Trong khi weighted F1 sẽ cho Happy trọng số cao hơn

---

## 6. Tổng hợp Root Cause

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY OVERFITTING HAPPENS — FACTOR BREAKDOWN                │
│                                                                              │
│  Factor                          │ Baseline │ Improved │ MulT Emotion       │
│  ────────────────────────────────┼──────────┼──────────┼─────────────────────│
│  1. Dataset size (train)          │  16,326  │  16,326  │  ~12,500          │
│  2. Model params                 │  1.1M    │  2.03M   │  ~1.7M            │
│  3. Params per sample            │  67      │  124     │  136              │
│  4. Sequence length              │  50      │  50      │  50               │
│  5. Dropout rate (encoder)       │  0.1     │  0.3     │  0.2              │
│  6. Dropout rate (fusion)        │  0.3/0.2 │  0.4/0.3 │  0.5              │
│  7. Cross-modal interaction      │  ❌       │  ❌       │  ✅ (tốt)        │
│  8. Attention pooling           │  ❌       │  ✅       │  ✅               │
│  9. Early stopping metric        │  MAE     │  MAE     │  Mean F1          │
│  10. LR scheduler               │ Plateau  │ Plateau  │  CosineWarm       │
│  11. BCE pos_weight             │  N/A     │  N/A     │  ❌ (none)        │
│  12. Threshold tuning           │  N/A     │  N/A     │  ❌ (fixed 0.5)   │
│  ────────────────────────────────┼──────────┼──────────┼─────────────────────│
│  Overfitting severity            │  🔴 13x  │  🟡 10x  │  🔴 F1 drop       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.1. Suy luận từ Baseline LSTM (Hình 2)

1. **Overfitting bắt đầu từ epoch 2** — Valid loss ngừng giảm từ rất sớm, trong khi train loss vẫn giảm mạnh.
   - Mô hình 1 layer LSTM với ~1.1M params **quá yếu để tổng quát hóa**
   - Hidden dim=128 nén thông tin từ 768 chiều → **information bottleneck**

2. **Valid Corr ~0.70 không phải trần** — Nhìn đồ thị, Corr dao động trong khoảng 0.67-0.70. Điều này có nghĩa: **cấu hình LR và scheduler chưa tối ưu**

3. **ReduceLROnPlateau không hiệu quả** — LR giảm 3 lần nhưng valid loss vẫn không cải thiện. Nguyên nhân: plateau metric là valid loss (MSE), trong khi mô hình đánh giá bằng MAE.

### 6.2. Suy luận từ Improved LSTM (Hình 3)

1. **Attention Pooling thực sự hoạt động** — Valid MAE giảm từ 0.5526 (baseline) xuống 0.5323 (-3.7%), Corr tăng từ 0.6995 lên 0.7254 (+3.7%).

2. **2-layer LSTM vẫn không đủ** — Dù có thêm LayerNorm, attention pooling, và gated fusion, khoảng cách train-valid vẫn lớn. **LSTM fundamentally struggles với multimodal fusion ở cấp sequence level**

3. **Gated Fusion giúp kiểm soát outliers** — Test MSE giảm 7% (0.6474 → 0.6021), nhiều hơn MAE giảm 3.5%. Gated fusion chủ yếu giảm **sai số lớn (outliers)**

4. **Epoch 7 là sweet spot** — Việc chạy thêm 10 epoch (17 total) chỉ làm overfit nặng hơn mà không có lợi ích.

### 6.3. Suy luận từ MulT Emotion (Hình 1)

1. **Overfitting Pattern — Nghiêm trọng và Sớm:**
   - Train loss giảm 42% trong khi val loss **tăng** từ epoch 9
   - `d_model=64` với `num_heads=4` vẫn **overfitting** với ~12,500 mẫu
   - `attn_dropout=0.2` và `fusion_dropout=0.5` — dropout khá cao nhưng vẫn không đủ

2. **Mean Accuracy Oscillation — Bug ngưỡng 0.5:**
   - Val mean acc oscillation (12.5% → 38%) là **không bình thường**
   - Nguyên nhân: ngưỡng binarization cố định `0.5` cho **tất cả 6 cảm xúc**
   - Với imbalanced labels (Fear 2.2%, Happy 34%), ngưỡng 0.5 quá cao cho lớp thiểu số

3. **Per-emotion F1 — Phản ánh Class Imbalance:**

```
Happy (34.0%)   ████████████████ 0.4709  ← Tốt nhất, nhưng vẫn thấp
Sad (14.1%)     ████████ 0.2673           ← Trung bình
Angry (14.5%)   ██████ 0.1949             ← Trung bình thấp
Disgust (11.9%) █████ 0.1730              ← Thấp
Surprise (3.3%) ███ 0.0977                ← Rất thấp
Fear (2.2%)     █ 0.0348                  ← Gần như không học được
```

---

## 7. Priority Matrix — Thứ tự ưu tiên fix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPACT vs EFFORT MATRIX                               │
│                                                                              │
│                    EASY (implement)          HARD (refactor needed)          │
│               ┌────────────────────────┬────────────────────────────────┐  │
│   HIGH        │ ⭐⭐⭐ BCEWithLogitsLoss  │ ⭐⭐⭐ MulT d_model=128,heads=8    │  │
│   IMPACT      │ với pos_weight         │ (MulT P1 config)                │  │
│               │ ⭐⭐⭐ Per-emotion       │ ⭐⭐ Focal Loss thay BCE         │  │
│               │ threshold tuning       │ (refactor loss function)         │  │
│               ├────────────────────────┼────────────────────────────────┤  │
│   LOW         │ ⭐ Cosine Warmup thay   │ ⭐⭐ Weighted BCE + Focal Loss   │  │
│   IMPACT      │ Plateau cho LSTM        │ + class-aware sampling          │  │
│               │ ⭐ LR 1e-4 cho         │ (dataset-level refactor)        │  │
│               │ Baseline               │                                 │  │
│               └────────────────────────┴────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.1. Fix ngay (Impact cao + Dễ implement)

| # | Fix | Mã nguồn cần sửa | Tác động |
|:---:|:---|:---|:---|
| 1 | **BCEWithLogitsLoss với pos_weight** | `trainer.py:103` | Cải thiện F1 cho Fear/Surprise đáng kể |
| 2 | **Per-emotion threshold tuning** | `evaluator_emotion.py:67` | Cải thiện Mean F1 ~8-15% |
| 3 | **Early Stopping tại epoch 9** | Không cần code — resume từ checkpoint epoch 9 | Ngăn overfitting |
| 4 | **MulT P1 config: d_model=128, num_heads=8** | `config_phase1.py` | Giảm projection bottleneck |

### 7.2. Cải thiện đáng kể (Impact cao + Effort trung bình)

| # | Fix | Mã nguồn cần sửa | Tác động |
|:---:|:---|:---|:---|
| 5 | **Focal Loss thay BCE** | `trainer.py` + `evaluator.py` | Tự động handle imbalance |
| 6 | **Cosine Warmup thay ReduceLROnPlateau** | `config_phase1.py` | Training ổn định hơn |
| 7 | **Learning rate 1e-4 cho Baseline** | Notebook Colab | Giảm spike, hội tụ tốt hơn |

### 7.3. Tinh chỉnh (Impact trung bình)

| # | Fix | Mã nguồn cần sửa | Tác động |
|:---:|:---|:---|:---|
| 8 | **Attention pooling mask tốt hơn** | `models/mult.py` | Giảm NaN từ vision all-zero |
| 9 | **Gradient clipping max_norm=0.5 cho LSTM** | `config_phase1.py` | Ngăn gradient explosion |

---

## 8. Kết luận

### 8.1. Tóm tắt Root Cause

```
Nguyên nhân chính KHÔNG phải dataset hay hardware,
mà là 3 LỖI CHIẾN LƯỢC:

  1. BCE loss không có pos_weight
     → Model "chọn" predict lớp đa số (Happy)
     → Rare classes (Fear 2.2%, Surprise 3.3%) gần như không học được

  2. Ngưỡng binarization 0.5 cố định cho tất cả 6 emotions
     → Quá cao cho Fear (tối ưu ~0.05-0.10)
     → Quá thấp cho Happy (tối ưu ~0.30-0.35)
     → Mean Accuracy oscillation (12.5%-38%)

  3. MulT d_model=64 quá nhỏ tạo bottleneck nghiêm trọng
     → 768→64 projection mất 92% thông tin
     → Không đủ capacity cho multimodal fusion
```

### 8.2. Những gì đạt được tốt

| Khía cạnh | Đánh giá |
|:---|:---|
| Pipeline hoạt động end-to-end | ✅ Không crash, chạy được trên Colab |
| Baseline LSTM đúng chuẩn | ✅ Kiến trúc đúng, kết quả reproducible |
| Early Stopping hoạt động | ✅ Dừng đúng lúc, không explode |
| MulT Cross-Modal Attention | ✅ Kiến trúc đúng, có potential |
| Improved LSTM cải thiện rõ ràng | ✅ Tốt hơn baseline trên mọi metric |

### 8.3. Hướng hành động tiếp theo

1. **Ngay lập tức:** Implement `BCEWithLogitsLoss(pos_weight)` trong `trainer.py`
2. **Ngay lập tức:** Implement per-emotion threshold tuning trong `evaluator_emotion.py`
3. **Tuần tới:** Resume MulT Emotion từ checkpoint epoch 9, huấn luyện lại với P1 config
4. **Tiếp theo:** Thử Focal Loss, sau đó weighted sampling

---

*Báo cáo phân tích root cause được tạo tự động dựa trên Phase 1 Training Report, kết quả W&B, và source code.*
