# Phase 1 — Báo Cáo Tổng Kết Các Mô Hình Đã Train

> Ngày tổng kết: 2026-06-09
> Nguồn: GCS Bucket `mer-data-bucket-kandesfx`, Checkpoint & Log files

---

## 1. Tổng Quan Checkpoints

| # | Tên File | Model | Task | Loss | Trạng thái |
|---|----------|-------|------|------|-------------|
| 1 | `best_model_mult.pt` | MulT (aligned) | Sentiment | MSE+L1 | ✅可用 |
| 2 | `best_model_mult_unaligned.pt` | MulT (unaligned) | Sentiment | MSE+L1 | ✅可用 |
| 3 | `best_model_improved_lstm.pt` | Improved LSTM | Sentiment | MSE | ✅可用 |
| 4 | `best_model_mult_emotion.pt` | MulT (P0) | Emotion (6-class) | BCE | ✅可用 |
| 5 | `best_model_mult_emotion_p1_focal.pt` | MulT (P1) | Emotion (6-class) | Focal Loss | ❌ DIVERGED |

---

## 2. Sentiment Regression — Kết Quả

### 2.1. MOSEI Benchmark Reference

| Model | MAE | Corr | Acc-2 | Acc-5 | Acc-7 |
|-------|-----|------|-------|-------|-------|
| **MMSA MulT (SOTA reference)** | **0.5593** | **0.7331** | **81.15%** | **54.18%** | **52.84%** |
| Improved LSTM | 0.5859 | 0.7229 | 81.37% | 51.53% | 49.71% |
| Baseline LSTM | 0.6071 | 0.6995 | 81.03% | 49.73% | 48.36% |

### 2.2. Mô Hình Đã Train (cần evaluate trên Colab)

**Chạy notebook `08_evaluate_all_models.ipynb` để lấy kết quả thực tế.**

| Model | File | Target MAE | Target Corr |
|-------|------|-----------|------------|
| MulT (aligned) | `best_model_mult.pt` | ≤ 0.5700 | ≥ 0.7300 |
| MulT (unaligned) | `best_model_mult_unaligned.pt` | ≤ 0.5700 | ≥ 0.7300 |
| Improved LSTM | `best_model_improved_lstm.pt` | ≤ 0.5900 | ≥ 0.7200 |

---

## 3. Emotion Classification — Kết Quả

### 3.1. Mô Hình P0 — MulT Emotion với BCE (ĐÃ TRAIN)

**Checkpoint:** `best_model_mult_emotion.pt`
**Cấu hình:** MulT, d_model=64, num_heads=4, fusion_hidden_dim=128, BCEWithLogitsLoss

**Kết quả (threshold cố định 0.5):**

| Split | Mean F1 | Mean Acc | Happy F1 | Sad F1 | Angry F1 | Surprise F1 | Disgust F1 | Fear F1 |
|-------|---------|----------|----------|--------|---------|------------|------------|---------|
| Valid | 0.2118 | — | — | — | — | — | — | — |
| **Test** | **0.2307** | — | **0.4709** | ~0.27 | ~0.19 | ~0.10 | ~0.17 | **0.0348** |

### 3.2. Mô Hình P0.1 — Threshold Tuning (ĐÃ TRAIN)

**Checkpoint:** `best_model_mult_emotion.pt` (cùng model, chỉ đổi threshold)

**Optimal Thresholds per Emotion:**

| Emotion | Optimal Threshold | Delta vs 0.5 |
|---------|------------------|---------------|
| Happy | 0.60 | +0.10 |
| Sad | 0.50 | +0.00 |
| Angry | 0.55 | +0.05 |
| Surprise | 0.50 | +0.00 |
| Disgust | 0.55 | +0.05 |
| Fear | 0.50 | +0.00 |

**Kết quả với Tuned Thresholds:**

| Split | Mean F1 | Mean Acc | Mean MAE |
|-------|---------|----------|----------|
| Valid | 0.2535 | 0.4954 | 1.4286 |
| **Test** | **0.2621** | **0.4739** | **1.4186** |

**Cải thiện so với P0 baseline:**
- Valid: +0.0417 Mean F1 (+19.7%)
- Test: +0.0315 Mean F1 (+13.6%)

**Chi tiết per-emotion (Test set):**

| Emotion | F1 (tuned) | Acc | MAE |
|---------|-----------|-----|-----|
| Happy | **0.5984** | 0.6991 | 1.3775 |
| Sad | 0.2395 | 0.2170 | 1.4049 |
| Angry | 0.3165 | 0.8121 | 1.3868 |
| Surprise | 0.0558 | 0.0443 | 1.4543 |
| Disgust | 0.3048 | 0.8332 | 1.4228 |
| Fear | 0.0576 | 0.2377 | 1.4656 |

### 3.3. Mô Hình P1 — MulT Emotion với Focal Loss (DIVERGED — KHÔNG DÙNG ĐƯỢC)

**Checkpoint:** `best_model_mult_emotion_p1_focal.pt`
**Bug:** 2 lỗi trong `FocalLoss` (pos_weight nhân nhầm vào alpha_t + abnormal reduction)

**Loss theo epoch:**

| Epoch | Train Loss | Valid Loss |
|-------|-----------|-----------|
| 1 | 0.083 | 0.086 |
| 2 | -1.37 | -0.67 |
| 3 | -7.99 | -4.01 |
| 4 | -34.39 | -17.35 |
| 5 | -85.83 | -43.39 |
| 6 | -178.33 | -90.47 |
| 7 | -314.62 | -159.86 |
| 8 | -521.66 | -265.21 |

**Kết luận:** Gradient explosion từ epoch 2. Model không học được gì. **Cần retrain.**

---

## 4. Phân Tích Bug FocalLoss

### Root Cause

2 lỗi trong `training/losses/focal_loss.py`:

**Bug 1 — pos_weight nhân vào alpha_t:**
```python
# SAI: pos_weight (30-50) nhân trực tiếp vào alpha_t
alpha_t = pos_weight.unsqueeze(0) * targets + (1.0 - targets)
# → Với pos_weight=40, mỗi positive sample có weight gấp 40 lần
# → focal_loss = alpha_t * focal_weight * bce → inflated 40x

# ĐÚNG (sau fix): alpha chuẩn Lin et al. 2017
alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
```

**Bug 2 — Abnormal reduction:**
```python
# SAI: sum / mean_positives_per_sample
# → Inflated thêm khi batch có ít positives

# ĐÚNG (sau fix): standard mean
return focal_loss.mean()
```

### Fix đã commit

- `training/losses/focal_loss.py`: alpha_t dùng `alpha` chuẩn, reduction về `.mean()`
- `training/trainer.py`: `FocalLoss` dùng `pos_weight=None` (class imbalance đã được `gamma` xử lý)

---

## 5. Hướng Dẫn Evaluate Các Model

### Bước 1: Mở notebook `08_evaluate_all_models.ipynb` trên Colab

Notebook đã tạo tại: `notebooks/08_evaluate_all_models.ipynb`

### Bước 2: Chạy các cell theo thứ tự

1. Cell 1: Mount Google Drive
2. Cell 2: Clone/Pull repo
3. Cell 3: Download data từ GCS
4. Cell 4: **EVALUATE ALL MODELS** (cell mới)

### Bước 3: Xem kết quả

Output sẽ gồm:
- Bảng so sánh đầy đủ Sentiment (MAE, Corr, Acc-2/5/7)
- Bảng so sánh đầy đủ Emotion (Mean F1, per-emotion F1, MAE)
- File `all_evaluations.json` lưu về Drive

---

## 6. Bảng Tổng Hợp

### Sentiment Regression

| Model | Aligned | Status | Cần Evaluate |
|-------|---------|--------|-------------|
| MulT | ✅ | ✅ Train tốt | **Có — chạy notebook** |
| MulT (unaligned) | ❌ | ✅ Train tốt | **Có — chạy notebook** |
| Improved LSTM | ✅ | ✅ Train tốt | **Có — chạy notebook** |

### Emotion Classification

| Model | Config | Status | Mean F1 Test | Cần Evaluate |
|-------|--------|--------|-------------|-------------|
| MulT (P0) | d=64, h=4, BCE | ✅ | 0.2307 (raw) / 0.2621 (tuned) | **Có — chạy notebook** |
| MulT (P1) | d=128, h=8, Focal | ❌ Diverged | N/A | **Cần retrain với fix** |

---

## 7. Bước Tiếp Theo

### Ngay lập tức
- [ ] Chạy `08_evaluate_all_models.ipynb` trên Colab để lấy kết quả sentiment đầy đủ

### Ngắn hạn
- [ ] Retrain emotion P1 với Focal Loss đã fix (sau commit `1a15119`)
- [ ] Evaluate emotion P0 trên Colab với threshold tuning đầy đủ

### Trung hạn
- [ ] Compare sentiment models vs MMSA benchmark
- [ ] Chọn best model để tích hợp vào Emotion Data Studio
