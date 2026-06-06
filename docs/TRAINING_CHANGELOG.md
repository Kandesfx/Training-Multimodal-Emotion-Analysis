# MulT Training Changelog

Tài liệu này ghi lại lịch sử thay đổi, lý do, và kết quả mỗi lần tinh chỉnh mô hình MulT.

---

## Run 1: MulT Baseline (2026-06-06)

**Commit:** `110f05e` | **Notebook:** `03_mult_training.ipynb` | **Data:** `aligned_50.pkl`

### Hyperparameters
| Parameter | Value |
|:---|:---|
| d_model | 64 |
| num_heads | 4 |
| num_cross_layers | 4 |
| num_self_layers | 2 |
| ffn_dim | 128 |
| attn_dropout | 0.1 |
| fusion_hidden_dim | 128 |
| fusion_dropout | 0.3 |
| batch_size | 32 |
| learning_rate | 1e-4 |
| weight_decay | 1e-3 |
| loss | MSE only |
| scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |

### Results
| Metric | Train | Valid | Test | Target |
|:---|:---:|:---:|:---:|:---:|
| Loss | 0.351 | 0.561 | 0.614 | — |
| MAE | 0.446 | 0.546 | **0.5751** | ≤ 0.5700 |
| Corr | 0.856 | 0.704 | **0.7196** | ≥ 0.7300 |
| Acc-2 | 81.6% | 78.9% | 78.5% | — |
| Acc-5 | 62.4% | 53.9% | 53.5% | — |
| Acc-7 | 59.7% | 52.8% | 52.1% | — |
| F1 | 0.823 | 0.798 | 0.794 | — |

**Best epoch:** 11 (early stop at 21, patience=10)

### Analysis
1. **Overfitting nghiêm trọng** — train/valid loss gap = 60%
2. **MAE chỉ cách target 0.0051** — rất gần ngưỡng
3. **LR giảm quá nhanh** — ReduceLROnPlateau kéo xuống 1e-5 sau ~15 epoch
4. **Loss MSE** không trực tiếp tối ưu metric mục tiêu (MAE)
5. **Dropout quá thấp** cho mức capacity của model

### Bugs đã phát hiện & fix
- `loss=nan` do 482 vision samples all-zero → fix `_ensure_valid_mask()`
- `SyntaxError` trong notebook JSON (unterminated string literal)

---

## Run 2: Optimization (2026-06-07) — Pending

**Commit:** _pending_ | **Notebook:** `03_mult_training.ipynb` | **Data:** `aligned_50.pkl`

### Changes from Run 1
| Parameter | Run 1 | Run 2 | Lý do |
|:---|:---|:---|:---|
| loss_type | mse | **mse_l1** | Trực tiếp tối ưu MAE (metric mục tiêu) |
| l1_weight | — | **0.5** | 50% MSE + 50% L1 (balanced) |
| scheduler_type | plateau | **cosine_warmup** | Transformer cần warmup, cosine decay tốt hơn |
| warmup_epochs | — | **5** | 5 epoch đầu LR tăng dần từ 1% → 100% |
| min_lr | — | **1e-6** | Cosine decay floor |
| attn_dropout | 0.1 | **0.2** | Giảm overfitting tầng attention |
| fusion_dropout | 0.3 | **0.5** | Giảm overfitting tầng fusion head |
| weight_decay | 1e-3 | **5e-3** | Regularization mạnh hơn 5× |

### Expected Impact
1. **MSE+L1 loss** → giảm MAE 0.01–0.02 (model trực tiếp minimize absolute error)
2. **Cosine warmup** → training ổn định hơn, LR không giảm quá sớm
3. **Tăng regularization** → thu hẹp train/valid gap (60% → ~20–30%)

### Results
_Chưa chạy — update sau khi train trên Colab_

| Metric | Train | Valid | Test | Target | vs Run 1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| MAE | — | — | — | ≤ 0.5700 | — |
| Corr | — | — | — | ≥ 0.7300 | — |
