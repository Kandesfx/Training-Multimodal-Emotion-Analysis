# MulT Training Changelog

Tài liệu này ghi lại lịch sử thay đổi, lý do, và kết quả mỗi lần tinh chỉnh mô hình MulT.

---

## Run 1: MulT Aligned Baseline (2026-06-06)

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

**Best epoch:** 11 (early stop at 21)

### Analysis
1. **Overfitting nghiêm trọng** — train/valid loss gap = 60%
2. **MAE chỉ cách target 0.0051**
3. **LR giảm quá nhanh** — ReduceLROnPlateau kéo xuống 1e-5
4. **Loss MSE** không trực tiếp tối ưu MAE

### Bugs đã fix
- `loss=nan` do 482 vision samples all-zero → `_ensure_valid_mask()`
- `SyntaxError` trong notebook JSON

---

## Run 2: MulT Aligned Optimized (2026-06-07)

**Commit:** `f8c19a1` | **Notebook:** `03_mult_training.ipynb` | **Data:** `aligned_50.pkl`  
**wandb:** https://wandb.ai/kandesfx-kandesfx/bcda-phase1/runs/6liarpgf

### Changes from Run 1
| Parameter | Run 1 | Run 2 | Lý do |
|:---|:---|:---|:---|
| loss_type | mse | **mse_l1** | Trực tiếp tối ưu MAE |
| l1_weight | — | **0.5** | 50% MSE + 50% L1 |
| scheduler_type | plateau | **cosine_warmup** | Warmup ổn định Transformer |
| warmup_epochs | — | **5** | LR tăng dần 5 epoch đầu |
| min_lr | — | **1e-6** | Cosine decay floor |
| attn_dropout | 0.1 | **0.2** | Giảm overfitting |
| fusion_dropout | 0.3 | **0.5** | Giảm overfitting |
| weight_decay | 1e-3 | **5e-3** | Regularization 5× |

### Test Results
| Metric | Train (ep17) | Valid (ep17) | Test (ep17) | Target | vs Run 1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Loss | 0.435 | 0.538 | 0.586 | — | ↓ 0.028 |
| MAE | 0.463 | 0.534 | **0.5687** | ≤ 0.5700 ✅ | ↓ 0.006 |
| Corr | 0.834 | 0.706 | **0.7281** | ≥ 0.7300 ❌ | ↑ 0.009 |
| Acc-2 | 86.3% | 83.9% | 80.7% | — | ↑ 2.2% |
| Acc-5 | 60.8% | 54.5% | 54.2% | — | ↑ 0.7% |
| Acc-7 | 58.4% | 53.4% | 52.7% | — | ↑ 0.6% |
| F1 | 0.856 | 0.844 | 0.813 | — | ↑ 0.019 |

**Best epoch:** 17 (early stop at 27)

### Analysis
1. ✅ **MAE = 0.5687 — PASS target** (Run 1: 0.5751)
2. ❌ **Corr = 0.7281 — thiếu 0.0019** (nhưng cải thiện +0.009 từ Run 1)
3. ✅ **Overfitting giảm mạnh** — 22% (vs 60%)
4. ✅ **Best epoch 17** (vs 11) — cosine warmup giúp converge lâu hơn

---

## Run 3: MulT Unaligned (2026-06-07)

**Commit:** `f8c19a1` | **Notebook:** `04_mult_unaligned_training.ipynb` | **Data:** `unaligned_50.pkl`  
**wandb:** https://wandb.ai/kandesfx-kandesfx/bcda-phase1/runs/gfqb1kgz

### Hyperparameters
Same as Run 2 + `unaligned_50.pkl` (audio/vision seq_len=500)

### Test Results
| Metric | Train (ep18) | Valid (ep18) | Test (ep18) | Target | vs Run 2 |
|:---|:---:|:---:|:---:|:---:|:---:|
| Loss | 0.423 | 0.535 | 0.581 | — | ↓ 0.005 |
| MAE | 0.460 | 0.533 | **0.5641** | ≤ 0.5700 ✅ | ↓ 0.005 |
| Corr | 0.838 | 0.711 | **0.7200** | ≥ 0.7300 ❌ | ↓ 0.008 |
| Acc-2 | 85.0% | 82.4% | **82.6%** | — | ↑ 1.9% |
| Acc-5 | 61.9% | 54.5% | 54.3% | — | — |
| Acc-7 | 59.6% | 53.4% | 52.8% | — | — |
| F1 | 0.853 | 0.832 | 0.825 | — | ↑ 0.012 |

**Best epoch:** 18 (early stop at 28)

### Analysis
1. ✅ **MAE = 0.5641 — best MAE across all models** (PASS target)
2. ❌ **Corr = 0.7200 — thấp hơn aligned** (0.7200 vs 0.7281)
3. ✅ **Acc-2 = 82.6% — best Acc-2 across all models**
4. ✅ **Overfitting 26%** — kiểm soát tốt

---

## Tổng hợp toàn bộ Phase 1

| # | Model | Test MAE ↓ | Test Corr ↑ | Test Acc-2 ↑ | Test F1 ↑ | Best Ep |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | Baseline LSTM | 0.6071 | 0.6995 | 81.0% | — | — |
| 2 | Improved LSTM | 0.5859 | 0.7229 | 81.4% | — | — |
| 3 | MulT Aligned (Run 1) | 0.5751 | 0.7196 | 78.5% | 0.794 | 11 |
| 4 | **MulT Aligned (Run 2)** | 0.5687 | **0.7281** | 80.7% | 0.813 | 17 |
| 5 | **MulT Unaligned** | **0.5641** | 0.7200 | **82.6%** | **0.825** | 18 |

### Target Status
| Target | Best Model | Value | Status |
|:---|:---|:---:|:---:|
| MAE ≤ 0.5700 | MulT Unaligned | **0.5641** | ✅ PASS |
| Corr ≥ 0.7300 | MulT Aligned Run 2 | **0.7281** | ❌ FAIL (-0.0019) |
