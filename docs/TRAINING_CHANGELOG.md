# MulT Training Changelog

Tài liệu này ghi lại lịch sử thay đổi, lý do, và kết quả mỗi lần tinh chỉnh mô hình MulT.

---

## Run 1: MulT Baseline (2026-06-06)

**Commit:** `110f05e` | **Notebook:** `03_mult_training.ipynb` | **Data:** `aligned_50.pkl`  
**wandb:** https://wandb.ai/kandesfx-kandesfx/bcda-phase1/runs/2jdihi1a

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

### Test Results
| Metric | Train | Valid | Test | Target |
|:---|:---:|:---:|:---:|:---:|
| Loss | 0.351 | 0.561 | 0.614 | — |
| MAE | 0.446 | 0.546 | **0.5751** | ≤ 0.5700 ❌ |
| Corr | 0.856 | 0.704 | **0.7196** | ≥ 0.7300 ❌ |
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

## Run 2: MulT Aligned Optimized (2026-06-07)

**Commit:** `f8c19a1` | **Notebook:** `03_mult_training.ipynb` | **Data:** `aligned_50.pkl`  
**wandb:** https://wandb.ai/kandesfx-kandesfx/bcda-phase1/runs/6liarpgf

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

### Test Results
| Metric | Train (ep17) | Valid (ep17) | Test (ep17) | Target | vs Run 1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Loss | 0.435 | 0.538 | 0.586 | — | ↓ 0.028 |
| MAE | 0.463 | 0.534 | **0.5687** | ≤ 0.5700 ✅ | ↓ 0.0064 |
| Corr | 0.834 | 0.706 | **0.7281** | ≥ 0.7300 ❌ | ↑ 0.0085 |
| Acc-2 | 86.3% | 83.9% | **80.7%** | — | ↑ 2.2% |
| Acc-5 | 60.8% | 54.5% | 54.2% | — | ↑ 0.7% |
| Acc-7 | 58.4% | 53.4% | 52.7% | — | ↑ 0.6% |
| F1 | 0.856 | 0.844 | 0.813 | — | ↑ 0.019 |

**Best epoch:** 17 (early stop at 27, patience=10)

### Analysis
1. ✅ **MAE = 0.5687 — PASS target ≤ 0.5700** (Run 1 was 0.5751)
2. ❌ **Corr = 0.7281 — chỉ thiếu 0.0019** so với target 0.7300
3. ✅ **Overfitting giảm** — train/valid loss gap ~22% (vs Run 1's 60%)
4. ✅ **Cosine warmup hiệu quả** — best epoch 17 vs 11, model converge tốt hơn
5. ✅ **Combined loss** trực tiếp cải thiện MAE đúng như dự đoán

---

## Run 3: MulT Unaligned (2026-06-07)

**Commit:** `f8c19a1` | **Notebook:** `04_mult_unaligned_training.ipynb` | **Data:** `unaligned_50.pkl`

### Hyperparameters
Same as Run 2 (mse_l1, cosine_warmup, dropout 0.2/0.5, weight_decay 5e-3)  
+ Data: unaligned_50.pkl (audio/vision seq_len=500, text seq_len=50)

### Validation Results (best epoch 18)
| Metric | Train (ep18) | Valid (ep18) |
|:---|:---:|:---:|
| Loss | 0.423 | 0.535 |
| MAE | 0.460 | **0.5327** |
| Corr | 0.838 | **0.7108** |
| Acc-2 | 85.0% | 82.4% |
| Acc-5 | 61.9% | 54.5% |
| Acc-7 | 59.6% | 53.4% |

**Best epoch:** 18 (early stop at 28, patience=10)

### Analysis
1. ✅ **Valid MAE = 0.5327 — vượt target rõ ràng** (best MAE trong tất cả models)
2. ❌ **Valid Corr = 0.7108 — thấp hơn aligned** (0.7108 vs 0.7281)
3. ✅ **Overfitting kiểm soát tốt** — train/valid gap ~26%
4. ⚠️ Test metrics chưa có trong GCS history (nằm trong Colab cell output)

---

## Tổng hợp tất cả models

| Model | Test MAE ↓ | Test Corr ↑ | Test Acc-2 ↑ | Best Epoch |
|:---|:---:|:---:|:---:|:---:|
| Baseline LSTM | 0.6071 | 0.6995 | 81.0% | — |
| Improved LSTM | 0.5859 | 0.7229 | 81.4% | — |
| MulT Aligned Run 1 | 0.5751 | 0.7196 | 78.5% | 11 |
| **MulT Aligned Run 2** | **0.5687** | **0.7281** | **80.7%** | 17 |
| MulT Unaligned | 0.5327* | 0.7108* | 82.4%* | 18 |

*\* Valid metrics (test metrics pending)*

**Target: MAE ≤ 0.5700 ✅ (Run 2), Corr ≥ 0.7300 ❌ (thiếu 0.0019)**
