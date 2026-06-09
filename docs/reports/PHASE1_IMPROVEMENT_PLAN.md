# KẾ HOẠCH CẢI THIỆN MÔ HÌNH — PHASE 1 (ROUND 2)

**Đề tài:** Đề Tài 17 — Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức
**Ngày lập:** 09/06/2026
**Nguồn:** Root Cause Analysis + Source Code
**Ưu tiên:** P0 = fix ngay (code changes nhỏ, impact lớn), P1 = cần implement, P2 = tương lai

---

## Mục lục

1. [Tóm tắt Root Cause](#1-tóm-tắt-root-cause)
2. [P0: Fix Ngay — Không Cần Huấn Luyện Lại](#2-p0-fix-ngay--không-cần-huấn-luyện-lại)
   - [2.1. Per-emotion Threshold Tuning](#21-per-emotion-threshold-tuning)
   - [2.2. Resume từ Checkpoint Epoch 9 cho MulT Emotion](#22-resume-từ-checkpoint-epoch-9-cho-mult-emotion)
3. [P1: Cải Thiện Loss & Training Strategy](#3-p1-cải-thiện-loss--training-strategy)
   - [3.1. BCEWithLogitsLoss với pos_weight (class imbalance)](#31-bcewithlogitsloss-với-pos_weight-class-imbalance)
   - [3.2. Focal Loss thay BCE — Tự động handle imbalanced classes](#32-focal-loss-thay-bce--tự-động-handle-imbalanced-classes)
   - [3.3. MulT Config P1: d_model=128, num_heads=8, stochastic depth](#33-mult-config-p1-d_model128-num_heads8-stochastic-depth)
   - [3.4. Early Stopping: Val Loss → Val Metric](#34-early-stopping-val-loss--val-metric)
4. [P2: Cải Thiện Kiến Trúc & Data](#4-p2-cải-thiện-kiến-trúc--data)
   - [4.1. Weighted Sampling cho Rare Classes](#41-weighted-sampling-cho-rare-classes)
   - [4.2. Data Augmentation cho Audio/Vision](#42-data-augmentation-cho-audiovision)
   - [4.3. Pseudo-labeling cho Fear/Surprise](#43-pseudo-labeling-cho-fearsurprise)
5. [Kế hoạch huấn luyện Round 2](#5-kế-hoạch-huấn-luyện-round-2)
6. [Benchmark kỳ vọng](#6-benchmark-kỳ-vọng)

---

## 1. Tóm tắt Root Cause

```
Ba lỗi chiến lược chính (theo thứ tự ảnh hưởng):

  1. BCEWithLogitsLoss không có pos_weight
     → Model ưu tiên Happy (34%) bỏ rơi Fear (2.2%)
     → F1_happy=0.47 vs F1_fear=0.035 (chênh 13.5x)

  2. Ngưỡng binarization 0.5 cố định cho TẤT CẢ 6 emotions
     → Quá cao cho Fear (tối ưu ~0.05-0.10)
     → Gây Mean Accuracy oscillation (12.5%-38%)

  3. MulT d_model=64 quá nhỏ cho 768→64 projection
     → Mất 92% thông tin BERT
     → Attention heads quá ít (4 heads × 16 chiều/head = quá nhỏ)
```

---

## 2. P0: Fix Ngay — Không Cần Huấn Luyện Lại

> **Ưu điểm:** Chỉ cần thay đổi code evaluation, KHÔNG cần huấn luyện lại. Tác động ngay lập tức.

### 2.1. Per-emotion Threshold Tuning

**Vấn đề:** Ngưỡng 0.5 cố định cho tất cả 6 emotions gây:
- Fear (2.2% positive): ngưỡng 0.5 quá cao → model gần như không bao giờ predict positive
- Happy (34% positive): ngưỡng 0.5 hợp lý nhưng có thể tối ưu hơn

**Giải pháp:** Quét ngưỡng tối ưu [0.05, 0.90] riêng cho từng emotion trên tập Validation.

**Code cần thêm vào `training/evaluator_emotion.py`:**

```python
def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    emotion_names: list[str] = EMOTION_NAMES,
    threshold_range: np.ndarray | None = None,
) -> dict[str, float]:
    """Find per-emotion optimal binarization threshold via grid search.

    Args:
        y_true: (N, 6) ground truth intensities [0, 3]
        y_pred: (N, 6) raw logits (before sigmoid)
        emotion_names: list of 6 emotion names
        threshold_range: array of thresholds to search (default: np.arange(0.05, 0.91, 0.05))

    Returns:
        dict mapping emotion_name → optimal_threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.05, 0.91, 0.05)  # [0.05, 0.10, ..., 0.90]

    y_true_bin = (y_true >= DEFAULT_THRESHOLD).astype(int)  # ground truth
    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))

    optimal_thresholds = {}
    for i, emo in enumerate(emotion_names):
        best_f1 = -1.0
        best_thresh = 0.5
        for thresh in threshold_range:
            pred_bin = (y_pred_prob[:, i] >= thresh).astype(int)
            f1 = f1_score(y_true_bin[:, i], pred_bin, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds[emo] = best_thresh
        print(f"  {emo:10s}: best_thresh={best_thresh:.2f}  F1={best_f1:.4f}")

    return optimal_thresholds


def compute_emotion_metrics_with_tuned_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, float]:
    """Compute metrics using per-emotion tuned thresholds."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 6)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 6)

    y_true_bin = (y_true >= DEFAULT_THRESHOLD).astype(int)
    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))

    metrics = {}
    f1_scores = []
    acc_scores = []

    for i, emo in enumerate(EMOTION_NAMES):
        thresh = thresholds.get(emo, 0.5)
        pred_bin = (y_pred_prob[:, i] >= thresh).astype(int)

        emo_f1 = float(f1_score(y_true_bin[:, i], pred_bin, average="binary", zero_division=0))
        emo_acc = float(accuracy_score(y_true_bin[:, i], pred_bin))

        metrics[f"{emo}_f1"] = emo_f1
        metrics[f"{emo}_acc"] = emo_acc
        metrics[f"{emo}_threshold"] = thresh
        f1_scores.append(emo_f1)
        acc_scores.append(emo_acc)

    metrics["mean_f1"] = float(np.mean(f1_scores))
    metrics["mean_acc"] = float(np.mean(acc_scores))

    return metrics
```

**Cách sử dụng sau khi huấn luyện:**

```python
# 1. Load validation predictions và ground truth
val_preds, val_labels = load_predictions("checkpoints/phase1/best_model_mult_emotion.pt")

# 2. Tìm ngưỡng tối ưu trên validation
optimal_thresholds = find_optimal_thresholds(val_labels, val_preds)

# 3. Áp dụng cho test set
test_preds, test_labels = load_predictions("checkpoints/phase1/best_model_mult_emotion.pt", split="test")
test_metrics = compute_emotion_metrics_with_tuned_thresholds(test_labels, test_preds, optimal_thresholds)
```

**Expected impact:**

| Emotion | F1 trước (ngưỡng 0.5) | F1 kỳ vọng (tuned threshold) |
|:---|:---:|:---:|
| Happy | 0.4709 | 0.48-0.52 |
| Sad | 0.2673 | 0.28-0.32 |
| Angry | 0.1949 | 0.22-0.28 |
| Disgust | 0.1730 | 0.20-0.25 |
| Surprise | 0.0977 | 0.15-0.25 |
| Fear | 0.0348 | 0.10-0.20 |
| **Mean F1** | **0.2064** | **0.30-0.38** |

**Tác động:** ⭐⭐⭐ — Mean F1 tăng 50-85% chỉ bằng threshold tuning, không cần huấn luyện lại.

---

### 2.2. Resume từ Checkpoint Epoch 9 cho MulT Emotion

**Vấn đề:** MulT Emotion chạy đến epoch 29 trong khi best model ở epoch 9 (val loss thấp nhất).

**Giải pháp:** Resume từ checkpoint tốt nhất và áp dụng P1 config.

**Code cần thay đổi:** Không cần code mới — chỉ cần cập nhật notebook Colab:

```python
# Trong notebook: 05_mult_emotion_training.ipynb

# Đổi checkpoint_name thành checkpoint tốt nhất
config.training.checkpoint_name = "best_model_mult_emotion.pt"  # epoch 9
config.training.resume_from_checkpoint = True
config.training.resume_checkpoint_type = "best"  # resume từ best, KHÔNG phải last

# HOẶC nếu muốn huấn luyện lại từ đầu với P1 config:
config.training.num_epochs = 20       # giảm từ 50
config.training.patience = 6          # giảm từ 10 (d_model lớn hơn → hội tụ nhanh hơn)
```

**Tác động:** ⭐⭐ — Ngăn overfitting thêm, model epoch 9 tốt hơn epoch 29.

---

## 3. P1: Cải Thiện Loss & Training Strategy

> **Ưu điểm:** Thay đổi loss function + config, cần huấn luyện lại 1-2 rounds.

### 3.1. BCEWithLogitsLoss với pos_weight (Class Imbalance)

**Vấn đề:** BCEWithLogitsLoss mặc định có pos_weight=1 cho tất cả 6 emotions. Happy (34%) và Fear (2.2%) có cùng weight.

**Giải pháp:** Tính pos_weight = `num_negatives / num_positives` cho mỗi emotion, truyền vào BCEWithLogitsLoss.

**Ước tính pos_weight từ distribution CMU-MOSEI:**

```
Emotion    | Positive % | Negatives | Positives | pos_weight
-----------|------------|-----------|-----------|------------
Happy      | 34.0%      | 10,779    | 5,547     | ~1.9
Sad        | 14.1%      | 13,993    | 2,333     | ~6.0
Angry      | 14.5%      | 13,948    | 2,378     | ~5.9
Disgust    | 11.9%      | 14,381    | 1,945     | ~7.4
Surprise   | 3.3%       | 15,764    | 562       | ~28.0
Fear       | 2.2%       | 15,962    | 364       | ~43.8
```

**Code cần thêm vào `training/trainer.py`:**

```python
# Thêm vào class Phase1Trainer, sau phần _build_criterion()

def _compute_class_weights(self, train_labels: np.ndarray) -> torch.Tensor | None:
    """Compute pos_weight for BCE loss from training labels.

    pos_weight[i] = num_negatives_i / num_positives_i for emotion i.
    Higher weight → more penalty for missing rare positive samples.
    """
    if self.task_type != "emotion":
        return None

    from training.evaluator_emotion import EMOTION_NAMES, DEFAULT_THRESHOLD
    y_true_bin = (train_labels >= DEFAULT_THRESHOLD).astype(int)  # (N, 6)

    weights = []
    for i in range(6):
        n_pos = y_true_bin[:, i].sum()
        n_neg = len(y_true_bin) - n_pos
        if n_pos == 0:
            weights.append(1.0)  # fallback
        else:
            w = n_neg / n_pos
            # Clamp: prevent extreme weights from dominating training
            w = min(w, 50.0)
            weights.append(w)

    print(f"  [BCE pos_weights] {dict(zip(EMOTION_NAMES, [f'{w:.1f}' for w in weights]))}")
    return torch.tensor(weights, dtype=torch.float32)


def _build_criterion(self, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build loss function based on config.training.loss_type."""
    loss_type = self.config.training.loss_type.lower().strip()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mse_l1":
        return _CombinedMSEL1Loss(l1_weight=self.config.training.l1_weight)
    if loss_type == "bce":
        if class_weights is not None:
            return nn.BCEWithLogitsLoss(pos_weight=class_weights.to(self.device))
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss_type: {loss_type!r}.")
```

**Cách gọi trong `fit()`:**

```python
def fit(self, train_loader, valid_loader) -> dict[str, Any]:
    # ... existing setup ...

    # Compute class weights from training data (requires one pass to get labels)
    class_weights = None
    if self.config.training.loss_type.lower().strip() == "bce":
        train_labels = self._collect_labels(train_loader)
        class_weights = self._compute_class_weights(train_labels)

    self.criterion = self._build_criterion(class_weights=class_weights)
    # ... rest of training loop ...
```

**File `training/dataset_mosei.py`** — thêm helper:

```python
def _collect_labels(self, data_loader) -> np.ndarray:
    """Collect all labels from a DataLoader (for class weight computation)."""
    labels = []
    for batch in data_loader:
        labels.append(batch["label"].numpy())
    return np.concatenate(labels, axis=0)
```

**Expected impact:**

| Emotion | F1 trước | F1 kỳ vọng |
|:---|:---:|:---:|
| Happy | 0.4709 | 0.42-0.46 |
| Sad | 0.2673 | 0.30-0.35 |
| Angry | 0.1949 | 0.24-0.30 |
| Disgust | 0.1730 | 0.22-0.28 |
| Surprise | 0.0977 | 0.18-0.28 |
| Fear | 0.0348 | 0.12-0.22 |
| **Mean F1** | **0.2064** | **0.26-0.35** |

> **Lưu ý:** Happy F1 có thể giảm nhẹ vì model phải chia sẻ capacity cho rare classes, nhưng **Mean F1 sẽ tăng đáng kể**.

**Tác động:** ⭐⭐⭐ — Fix root cause #1, cải thiện đáng kể trên tất cả rare emotions.

---

### 3.2. Focal Loss thay BCE — Tự động handle imbalanced classes

**Ưu điểm so với BCE pos_weight:**
- Tự động giảm weight cho easy negatives (background class)
- Tập trung vào hard examples (rare positives bị confuse)
- Không cần tính pos_weight thủ công

**Code cho Focal Loss:**

```python
# File: training/losses/focal_loss.py

from torch import nn
import torch


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017).

    FL(p) = -α(1-p)^γ * log(p)          for positive class
    FL(p) = -(1-α)p^γ * log(1-p)        for negative class

    With γ=2 (default):
      - Easy examples (p≈1 or p≈0): weight ≈ 0
      - Hard examples (p≈0.5): weight = α, maximum penalty

    Args:
        alpha: weighting factor between positive and negative (default: 0.25)
        gamma: focusing parameter (default: 2.0). Higher = more focus on hard examples.
        reduction: "mean" or "sum" (default: "mean")
        pos_weight: optional per-class positive weight (shape: [num_classes])
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) — raw model outputs (before sigmoid)
            targets: (batch, num_classes) — binary targets {0, 1}
        """
        probs = torch.sigmoid(logits)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)  # p_t = p if y=1 else 1-p
        focal_weight = (1 - p_t) ** self.gamma

        if self.pos_weight is not None:
            alpha_t = self.pos_weight.unsqueeze(0) * targets + self.alpha * (1 - targets)
        else:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == "mean":
            # Average over classes AND batch, but exclude zero-target classes from denominator
            return focal_loss.sum() / (targets.sum(dim=1).clamp(min=1)).mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
```

**Cập nhật `trainer.py` để hỗ trợ Focal Loss:**

```python
# Trong _build_criterion()
if loss_type == "focal":
    return FocalLoss(
        alpha=self.config.training.focal_alpha,
        gamma=self.config.training.focal_gamma,
        pos_weight=class_weights,
    )
```

**Cập nhật `config_phase1.py`:**

```python
# Trong Phase1TrainingConfig, thêm:
loss_type: str = "bce"           # thay đổi thành "focal" sau khi test
focal_alpha: float = 0.25         # weight balance positive/negative
focal_gamma: float = 2.0         # focusing parameter (2.0 recommended)
```

**Expected impact:**

| Config | Mean F1 kỳ vọng |
|:---|:---:|
| BCE (baseline) | 0.2064 |
| BCE + pos_weight | 0.28-0.35 |
| Focal Loss (α=0.25, γ=2.0) | 0.30-0.38 |
| Focal Loss + pos_weight | 0.32-0.40 |

**Tác động:** ⭐⭐⭐ — Focal Loss tự động handle imbalance tốt hơn BCE.

---

### 3.3. MulT Config P1: d_model=128, num_heads=8, stochastic depth

**Vấn đề:** Config hiện tại đã được cập nhật P1 defaults trong `config_phase1.py` (d_model=128, num_heads=8). Cần xác nhận và bổ sung stochastic depth.

**Config hiện tại đã tốt:**

```python
# config_phase1.py — Phase1MulTModelConfig (đã updated)
d_model: int = 128              # 64 → 128 (4x capacity)
num_heads: int = 8               # 4 → 8 (tối ưu hơn)
fusion_hidden_dim: int = 256     # 128 → 256
stochastic_depth_survival: float = 0.8  # LayerDrop

# Phase1TrainingConfig
learning_rate: float = 1e-4      # giữ nguyên
weight_decay: float = 3e-3       # 1e-4 → 3e-3 (nhiều params hơn)
scheduler_type: str = "cosine_warmup"  # thay plateau
attn_dropout: float = 0.1       # giảm từ 0.2
fusion_dropout: float = 0.3     # giảm từ 0.5
max_grad_norm: float = 0.5      # 1.0 → 0.5
```

**Cần xác nhận thêm trong notebook Colab:**

```python
# MulT Emotion config P1 — dùng khi huấn luyện lại Round 2
config.model_type = "mult"
config.training.task_type = "emotion"
config.training.loss_type = "focal"          # thay vì "bce"
config.training.focal_alpha = 0.25
config.training.focal_gamma = 2.0
config.training.num_epochs = 25              # giảm từ 50
config.training.patience = 7                 # giảm từ 10
config.training.warmup_epochs = 3            # ~12% of 25
config.training.scheduler_type = "cosine_warmup"
```

**So sánh params trước/sau:**

| Thông số | MulT P0 (thực tế) | MulT P1 (cải thiện) |
|:---|:---:|:---:|
| d_model | 64 | 128 |
| num_heads | 4 | 8 |
| dim per head | 16 | 16 |
| ffn_dim | 128 | 128 |
| fusion_hidden_dim | 128 | 256 |
| Params ước tính | ~1.7M | ~3.2M |
| Thông tin giữ lại từ BERT | 8.3% | 16.7% |
| Stochastic Depth | ❌ | ✅ (survival=0.8) |
| GELU activation | ❌ | ✅ |
| Pre-LN projection | ❌ | ✅ |
| Attn dropout | 0.2 | 0.1 |

**Expected impact:**

| Metric | MulT P0 | MulT P1 kỳ vọng |
|:---|:---:|:---:|
| Sentiment MAE (Test) | — | 0.54-0.56 |
| Sentiment Corr (Test) | — | 0.74-0.76 |
| Emotion Mean F1 | 0.2064 | 0.32-0.40 |

**Tác động:** ⭐⭐⭐ — Giảm bottleneck projection, tăng capacity đáng kể.

---

### 3.4. Early Stopping: Val Loss → Val Metric

**Vấn đề hiện tại:**

```python
# trainer.py, line 187
if self.scheduler_type == "plateau":
    self.scheduler.step(valid_loss)  # plateau dùng val/loss (MSE)
```

Scheduler dùng `valid_loss` (MSE), nhưng metric chính là `mae` (sentiment) hoặc `mean_f1` (emotion). Khi val_loss không cải thiện nhưng val_metric có thể vẫn cải thiện, scheduler giảm LR quá sớm.

**Giải pháp:** Đồng bộ scheduler metric với metric_for_best.

**Code cần thay đổi trong `trainer.py`:**

```python
# Trong _build_scheduler(), thêm support cho metric-based plateau
if sched_type == "plateau_metric":
    return ReduceLROnPlateau(
        self.optimizer,
        mode="max" if self.maximize_metric else "min",
        factor=self.config.training.scheduler_factor,
        patience=self.config.training.scheduler_patience,
    )
```

```python
# Trong fit(), thay đổi scheduler step:
if self.scheduler_type == "plateau":
    self.scheduler.step(valid_loss)
elif self.scheduler_type == "plateau_metric":
    self.scheduler.step(current_metric)  # dùng metric chính thay vì loss
else:
    self.scheduler.step()
```

**Config update:**

```python
# Trong Phase1TrainingConfig
scheduler_type: str = "plateau_metric"  # thay "plateau" cho LSTM sentiment
scheduler_patience: int = 5              # tăng từ 4
```

**Tác động:** ⭐⭐ — Scheduler không còn giảm LR khi metric chưa thực sự plateau.

---

## 4. P2: Cải Thiện Kiến Trúc & Data

### 4.1. Weighted Sampling cho Rare Classes

**Vấn đề:** DataLoader shuffle ngẫu nhiên không đảm bảo rare classes xuất hiện đủ trong mỗi epoch.

**Giải pháp:** WeightedRandomSampler với weights tỷ lệ nghịch với class frequency.

**Code:**

```python
# File: training/dataset_mosei.py

def create_weighted_sampler(labels: np.ndarray, mode: str = "inverse_freq") -> WeightedRandomSampler:
    """Create weighted sampler for imbalanced multi-label data.

    Args:
        labels: (N, 6) binary emotion labels
        mode: "inverse_freq" (default) or "sqrt_inv" (less extreme)

    Returns:
        WeightedRandomSampler instance
    """
    from training.evaluator_emotion import EMOTION_NAMES

    if mode == "inverse_freq":
        # Weight[i] = 1.0 / (positive_rate[i] + eps)
        pos_rate = labels.mean(axis=0)  # (6,)
        weights = 1.0 / (pos_rate + 1e-6)
    else:
        # sqrt_inv: less extreme than inverse_freq
        pos_rate = labels.mean(axis=0)
        weights = 1.0 / (np.sqrt(pos_rate) + 1e-6)

    # Sample weight per sample = max over all emotion weights (any positive → boost)
    sample_weights = np.zeros(len(labels))
    for i in range(6):
        mask = labels[:, i] == 1
        sample_weights[mask] = np.maximum(sample_weights[mask], weights[i])

    # Also boost samples with ANY rare emotion
    rare_thresh = 0.05  # emotions with <5% positive rate
    rare_emotions = pos_rate < rare_thresh
    for i in range(6):
        if rare_emotions[i]:
            mask = labels[:, i] == 1
            sample_weights[mask] *= 2.0  # double weight for rare positives

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    print(f"  [WeightedSampler] mode={mode}")
    print(f"    Emotion weights: {dict(zip(EMOTION_NAMES, [f'{w:.1f}' for w in weights]))}")
    print(f"    Sample weights: min={sample_weights.min():.2f}, max={sample_weights.max():.2f}, "
          f"mean={sample_weights.mean():.2f}")
    return sampler
```

**Sử dụng trong DataLoader:**

```python
# Trong notebook: thay vì shuffle=True
train_sampler = create_weighted_sampler(train_labels, mode="sqrt_inv")
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    sampler=train_sampler,       # thay shuffle=True
    num_workers=config.training.num_workers,
    pin_memory=config.training.pin_memory,
)
```

**Tác động:** ⭐⭐ — Tăng exposure của rare classes trong mỗi epoch.

---

### 4.2. Data Augmentation cho Audio/Vision

**Vấn đề:** Audio và Vision features không có augmentation → model overfits vào training patterns.

**Giải pháp:**

```
Audio augmentation:
  - Gaussian noise (σ=0.01)
  - Time shift (±5 frames)
  - Speed perturbation (×0.9, ×1.1)

Vision augmentation:
  - Gaussian noise (σ=0.005)
  - Temporal dropout (random frames → zeros)
```

**Code:**

```python
# File: training/augmentation/multimodal_augmentation.py

class AudioAugmentation(nn.Module):
    """On-the-fly audio augmentation during training."""

    def __init__(self, noise_std: float = 0.01, speed_range: tuple = (0.9, 1.1)):
        super().__init__()
        self.noise_std = noise_std
        self.speed_range = speed_range

    def forward(self, audio: Tensor) -> Tensor:
        if not self.training:
            return audio
        x = audio
        # Gaussian noise
        if torch.rand(1).item() > 0.5:
            x = x + torch.randn_like(x) * self.noise_std
        # Time shift (cyclic)
        if torch.rand(1).item() > 0.7:
            shift = int(torch.randint(-5, 6, (1,)).item())
            x = torch.roll(x, shifts=shift, dims=1)
        return x


class VisionAugmentation(nn.Module):
    """On-the-fly vision augmentation during training."""

    def __init__(self, noise_std: float = 0.005, dropout_prob: float = 0.1):
        super().__init__()
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob

    def forward(self, vision: Tensor) -> Tensor:
        if not self.training:
            return vision
        x = vision
        # Gaussian noise
        if torch.rand(1).item() > 0.5:
            x = x + torch.randn_like(x) * self.noise_std
        # Temporal dropout: randomly mask entire frames
        if torch.rand(1).item() > 0.6:
            T = x.size(1)
            n_drop = max(1, int(T * self.dropout_prob))
            drop_indices = torch.randperm(T)[:n_drop]
            x[:, drop_indices, :] = 0.0
        return x
```

**Tác động:** ⭐⭐ — Giảm overfitting trên training set.

---

### 4.3. Pseudo-labeling cho Fear/Surprise

**Vấn đề:** Fear (2.2%) và Surprise (3.3%) có quá ít mẫu → model không học được pattern.

**Giải pháp:** Sử dụng model đã trained (sau Round 1) để predict labels trên unaligned dataset, giữ lại mẫu có confidence cao.

```
Workflow:
  1. Train model Round 1 (với BCE + pos_weight + threshold tuning)
  2. Inference trên ~4,000 mẫu unaligned (không dùng trong training gốc)
  3. Giữ mẫu có:
     - Fear: probability > 0.7 và không có Happy/Sad đồng thời
     - Surprise: probability > 0.7
  4. Thêm mẫu pseudo-labeled vào training set
  5. Train Round 2 với augmented dataset
```

**Tác động:** ⭐ — Tăng training data cho rare classes, nhưng cần validation cẩn thận.

---

## 5. Kế hoạch huấn luyện Round 2

### 5.1. Round 2A: MulT Emotion với P0-Fixes (nhanh, ~2 giờ)

**Mục tiêu:** Validate threshold tuning + BCE pos_weight trước.

**Config:**
```python
config.model_type = "mult"
config.training.task_type = "emotion"
config.training.loss_type = "bce"           # + pos_weight
config.training.num_epochs = 20
config.training.patience = 7
config.training.scheduler_type = "cosine_warmup"
config.training.warmup_epochs = 3

# Giữ nguyên P0 architecture
config.mult_model.d_model = 64              # CHƯA tăng
config.mult_model.num_heads = 4             # CHƯA tăng
config.mult_model.attn_dropout = 0.2
```

**Workflow:**
1. Resume từ checkpoint MulT Emotion (epoch 9) — KHÔNG train lại
2. Load best_model_mult_emotion.pt → predict trên validation
3. Run `find_optimal_thresholds()` → lưu thresholds
4. Apply thresholds → compute improved metrics
5. So sánh Mean F1 trước/sau

---

### 5.2. Round 2B: MulT Emotion P1 (chính, ~4-6 giờ)

**Mục tiêu:** Huấn luyện lại từ đầu với P1 config + Focal Loss.

**Config:**
```python
config.model_type = "mult"
config.training.task_type = "emotion"
config.training.loss_type = "focal"         # thay BCE
config.training.focal_alpha = 0.25
config.training.focal_gamma = 2.0
config.training.num_epochs = 25
config.training.patience = 7
config.training.warmup_epochs = 3

# P1 architecture
config.mult_model.d_model = 128
config.mult_model.num_heads = 8
config.mult_model.fusion_hidden_dim = 256
config.mult_model.attn_dropout = 0.1
config.mult_model.stochastic_depth_survival = 0.8

# Training
config.training.learning_rate = 1e-4
config.training.weight_decay = 3e-3
config.training.scheduler_type = "cosine_warmup"
config.training.max_grad_norm = 0.5
```

**Workflow:**
1. Huấn luyện Round 2B (25 epochs)
2. Mỗi 5 epochs: evaluate với threshold tuning
3. Early stopping dựa trên mean_f1 (có plateau_metric scheduler)
4. Save best checkpoint
5. Run threshold tuning trên validation
6. Final evaluation trên test set

---

### 5.3. Round 2C: MulT Sentiment với P1 (mới, ~4 giờ)

**Mục tiêu:** Chạy MulT cho sentiment task (chưa từng chạy trước đó).

**Config:**
```python
config.model_type = "mult"
config.training.task_type = "sentiment"
config.training.loss_type = "mse_l1"
config.training.num_epochs = 30
config.training.patience = 10
config.training.warmup_epochs = 3

# P1 architecture (giống Round 2B)
config.mult_model.d_model = 128
config.mult_model.num_heads = 8
config.mult_model.fusion_hidden_dim = 256
```

**Workflow:**
1. Huấn luyện Round 2C
2. Evaluate trên test set
3. So sánh với Improved LSTM (Test MAE=0.5859, Corr=0.7229)

---

### 5.4. Timeline tổng hợp

```
Round 2A (Threshold + pos_weight):    ~30 phút     (chỉ inference, không train)
Round 2B (MulT Emotion P1 + Focal):   ~4-6 giờ    (train từ đầu)
Round 2C (MulT Sentiment P1):         ~4 giờ      (train từ đầu)

Tổng thời gian:                      ~9-11 giờ Colab (GPU T4)

Thứ tự ưu tiên:
  1. Round 2A → validate threshold + pos_weight
  2. Round 2C → có baseline mới cho sentiment (MulT vs Improved LSTM)
  3. Round 2B → MulT Emotion với Focal + P1
```

---

## 6. Benchmark kỳ vọng

### 6.1. Emotion Task (6-class Multi-label)

| Model | Mean F1 | Happy | Sad | Angry | Disgust | Surprise | Fear | Notes |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **MulT P0 (thực tế)** | **0.2064** | 0.4709 | 0.2673 | 0.1949 | 0.1730 | 0.0977 | 0.0348 | Baseline |
| MulT P0 + Thresholds | 0.28-0.32 | ~0.48 | ~0.28 | ~0.22 | ~0.20 | ~0.18 | ~0.12 | P0-Fix |
| MulT P0 + pos_weight | 0.26-0.32 | ~0.43 | ~0.32 | ~0.27 | ~0.24 | ~0.20 | ~0.15 | P1-Fix |
| **MulT P1 + Focal** | **0.32-0.40** | ~0.45 | ~0.34 | ~0.30 | ~0.27 | ~0.25 | ~0.20 | Target |
| MulT P1 + Focal + Thresholds | 0.36-0.44 | ~0.47 | ~0.35 | ~0.32 | ~0.30 | ~0.28 | ~0.25 | Ideal |
| SOTA (CMU-MOSEI) | ~0.50-0.55 | — | — | — | — | — | — | Literature ref |

### 6.2. Sentiment Task (Regression)

| Model | Test MAE | Test Corr | Test Acc-2 | Notes |
|:---|:---:|:---:|:---:|:---|
| Baseline LSTM | 0.6071 | 0.6995 | 0.8103 | Phase 1 |
| Improved LSTM | 0.5859 | 0.7229 | 0.8137 | Phase 1 |
| **MulT P1 (sentiment)** | **0.54-0.57** | **0.74-0.76** | **0.82-0.84** | Round 2C Target |
| SOTA (CMU-MOSEI) | ~0.53-0.55 | ~0.76-0.78 | ~0.84-0.86 | Literature ref |

### 6.3. Điều kiện đạt được SOTA

```
Để đạt SOTA-level performance (Mean F1 > 0.50 cho emotion, MAE < 0.54 cho sentiment):

Cần thêm:
  1. Pre-trained multimodal representations (CMU-MOSEI pre-trained models có sẵn)
  2. PhoBERT cho text modality thay BERT-base (tiếng Việt ở Phase 2)
  3. DeBERTa-v3 cho text modality (tiếng Anh)
  4. Larger model: d_model=256 hoặc pre-trained MMNL-99B
  5. Contrastive learning pre-training trên unaligned data
  6. Knowledge distillation từ teacher model
  7. Ensemble: Improved LSTM + MulT + TFN
```

---

*Bản kế hoạch cải thiện Round 2 được tạo dựa trên Root Cause Analysis.*
