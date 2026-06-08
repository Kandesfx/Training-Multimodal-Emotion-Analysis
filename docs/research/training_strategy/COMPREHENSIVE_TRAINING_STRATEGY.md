# Chiến Lược Training Tối Ưu Cho Multimodal Emotion Analysis

**Đề Tài 17: Hệ Thống Phân Tích Cảm Xúc Đa Phương Thức**

**Ngày soạn:** 2026-06-08
**Loại:** Chiến lược nghiên cứu & Kế hoạch triển khai
**Trạng thái:** Chờ phê duyệt

---

## Mục Lục

1. [Audit Hiện Trạng](#1-audit-hiện-trạng)
2. [Research: State-of-the-Art 2023-2024](#2-research-state-of-the-art-20232024)
3. [Comparison Table: Hiện tại vs Đề xuất](#3-comparison-table-hiện-tại-vs-đề-xuất)
4. [Kiến Trúc Đề Xuất Chi Tiết](#4-kiến-trúc-đề-xuất-chi-tiết)
5. [Implementation Roadmap](#5-implementation-roadmap)

---

## 1. Audit Hiện Trạng

### 1.1. Tổng Quan Pipeline

```
aligned_50.pkl / aligned_50_vi.pkl / unaligned_50.pkl
         │
         ▼
MOSEIAlignedDataset / MOSEIUnalignedDataset
         │  text (N,50,768) / audio (N,50,74) / vision (N,50,35)
         │  regression_labels [-3,+3]  /  emotion_labels (N,6)
         ▼
DataLoader (batch_size=32, shuffle_train)
         │
         ▼
MulTRegressor / ImprovedLSTMRegressor / EarlyFusionLSTMRegressor
         │
         ├── Projection: Linear → shared d_model
         ├── Positional Encoding (sinusoidal)
         ├── Cross-Modal Attention (6 flows: T←A, T←V, A←T, A←V, V←T, V←A)
         ├── Residual Merge
         ├── Self-Attention Transformer Encoder
         ├── Attention Pooling (MulT) / Last Hidden (LSTM)
         ├── LayerNorm
         └── Fusion Head: Concat → MLP → output
                  │
                  ▼
          _CombinedMSEL1Loss  (MSE+L1, l1_weight=0.5)
                  │
                  ▼
          AdamW (lr=1e-3, wd=1e-4)
          │
          ├── ReduceLROnPlateau / CosineWarmup
          ├── Gradient Clipping (max_norm=1.0)
          └── AMP (Automatic Mixed Precision)
```

### 1.2. Dataset Pipeline

| Thành phần | Chi tiết |
|:---|:---|
| **Aligned** | `aligned_50.pkl` — text/audio/vision đều `(N, 50, dim)`, word-aligned |
| **Unaligned** | `unaligned_50.pkl` — text `(N,50,768)`, audio/vision `(N,500,dim)` |
| **Vietnamese** | `aligned_50_vi.pkl` — English text dịch → PhoBERT encoding |
| **Emotion labels** | `(N, 6)` — happy, sad, angry, surprise, disgust, fear (0-3 scale) |
| **Matched mask** | Boolean mask, loại bỏ ~23% mẫu không khớp nhãn |
| **Train/Valid/Test** | 16,326 / 1,871 / 4,659 (sau mask: 12,484 / ~1,400 / ~3,500) |
| **Augmentation** | **KHÔNG CÓ** — raw features, no noise injection, no temporal masking |
| **Normalization** | **KHÔNG CÓ** — features pre-normalized từ MMSA |

### 1.3. Model Architectures

#### MulTRegressor (Multimodal Transformer)
```
Input: text (B,50,768), audio (B,50,74), vision (B,50,35)

Config: d_model=64, num_heads=4, num_cross_layers=4, num_self_layers=2,
        ffn_dim=128, attn_dropout=0.1, fusion_hidden_dim=128, fusion_dropout=0.3

Pipeline:
  1. Linear(768→64), Linear(74→64), Linear(35→64)
  2. Sinusoidal Positional Encoding
  3. 6 × CrossModalTransformerBlock (4 layers each) — 24 CrossModalAttentionLayer total
  4. Residual merge: T + T←A + T←V, etc.
  5. 3 × TransformerEncoder (2 layers each)
  6. AttentionPooling per modality → (B, 64) × 3
  7. LayerNorm per modality
  8. Concat → Linear(192→128) → LayerNorm → ReLU → Dropout(0.3)
                      → Linear(128→64) → ReLU → Dropout(0.15)
                      → Linear(64→1) → squeeze
  9. Output: (B,) sentiment hoặc (B,6) emotion
```

**Parameter count estimate:**
- Projections: `3 × (768+74+35) × 64 ≈ 176K`
- Cross-attention: `6 flows × 4 layers × (4×(64²/4) + 2×64×128 + 2×128×64) ≈ 1.2M`
- Self-attention: `3 encoders × 2 layers × (4×(64²/4) + 2×64×128 + 2×128×64) ≈ 300K`
- Attention pooling: `3 × (64×32 + 32×1) ≈ 6K`
- Fusion head: `192×128 + 128 + 128×64 + 64 + 64×1 ≈ 26K`
- **Total: ~1.7M parameters** (với d_model=64)

#### ImprovedLSTMRegressor
```
BiLSTM per modality → AttentionPooling → GatedFusion → MLP
text: 768→128 (2 layers, bidirectional) → 256
audio: 74→64 (bidirectional) → 128
vision: 35→64 (bidirectional) → 128
Concat → 3×projection_dim=384 → Gate network → Gated concat → MLP
```

#### EarlyFusionLSTMRegressor
```
BiLSTM per modality → Last hidden → Concat → MLP
text: 768→128 → 256
audio: 74→64 → 128
vision: 35→64 → 128
Concat 512 → FC(512→256) → BN → ReLU → Dropout(0.3)
               → FC(256→128) → ReLU → Dropout(0.2)
               → FC(128→1)
```

### 1.4. Loss Functions

| Task | Loss | Chi tiết |
|:---|:---|:---|
| **Sentiment (MSE)** | `nn.MSELoss()` | Đơn giản, nhưng MAE metric không được tối ưu trực tiếp |
| **Sentiment (MSE+L1)** | `(1-0.5)×MSE + 0.5×L1` | Tối ưu cả squared và absolute error |
| **Emotion (BCE)** | `nn.BCEWithLogitsLoss()` | Multi-label classification, sigmoid applied internally |

**Vấn đề:** Không có label smoothing, không có focal loss cho emotion (imbalanced classes: Happy 34% vs Fear 2.2%).

### 1.5. Training Loop

| Thành phần | Chi tiết |
|:---|:---|
| **Optimizer** | AdamW, lr=1e-3 (configurable) |
| **Schedulers** | `plateau` (ReduceLROnPlateau, factor=0.5, patience=3) hoặc `cosine_warmup` (Linear warmup → CosineAnnealing) |
| **Early stopping** | patience=8 epochs |
| **Gradient clipping** | max_norm=1.0 |
| **AMP** | torch.amp.GradScaler, enabled on CUDA |
| **Batch size** | 32 (aligned) / 16 (unaligned) |
| **Seed** | 42 |
| **Checkpoint** | Best + Last, upload to GCS automatically |
| **Metrics** | MAE (for model selection), Corr, Acc-2, Acc-5, Acc-7 |
| **WandB** | Optional, per-config |

### 1.6. Evaluation Metrics

**Sentiment (evaluator.py):**
- MAE, MSE, Corr (Pearson)
- Acc-2 (binary: ≥0 → positive)
- Acc-5 (-2,-1,0,1,2 rounded)
- Acc-7 (-3,-2,-1,0,1,2,3 rounded)
- F1 (binary weighted)

**Emotion (evaluator_emotion.py):**
- Per-emotion F1, Acc
- Mean F1, Mean Acc
- Per-emotion MAE, Mean MAE
- Threshold: 0.5 for binarization
- **Bug tiềm ẩn:** sigmoid probability × 3.0 để so sánh với ground truth 0-3 scale — điều này không chính xác vì sigmoid output range là (0,1) không phải (0,3)

### 1.7. Bottlenecks & Issues

#### 🔴 Critical Issues

**Issue 1: Model quá nhỏ — d_model=64 là bottleneck nghiêm trọng**

MulT với d_model=64 nén tất cả thông tin từ 768-dim text xuống 64-dim. Sau projection, mỗi modality chỉ có 64 dimensions để encode toàn bộ temporal dynamics và cross-modal interactions. Paper gốc MulT dùng d_model=40 (text không qua projection trực tiếp vào 40-dim), nhưng trên MOSEI, d_model=64-128 là phổ biến hơn. 64 vẫn chấp nhận được nhưng giới hạn capacity.

**Issue 2: Padding mask convention không nhất quán**

- `nn.MultiheadAttention`: `key_padding_mask=True` = ignored position (theo PyTorch convention)
- `AttentionPooling`: `mask=True` = valid position (tự định nghĩa)
- `CrossModalAttentionLayer`: nhận `key_padding_mask` nhưng convention không rõ ràng trong code

Trong `mult.py` hiện tại:
```python
t_mask = self._ensure_valid_mask(...)  # True = valid
a_mask = self._ensure_valid_mask(...)
v_mask = self._ensure_valid_mask(...)

# Trong cross-attention, truyền ~a_mask, ~v_mask, ~t_mask (invert):
t_with_a = self.cross_t_a(target=t, source=a, key_padding_mask=~a_mask)
```

Điều này đúng với PyTorch convention (True=ignored). Tuy nhiên, `CrossModalAttentionLayer.forward()` không validate mask convention và không có comment giải thích. Rủi ro cao khi maintain.

**Issue 3: Emotion evaluator — sigmoid scale mismatch**

Trong `evaluator_emotion.py`:
```python
y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))
y_pred_bin = (y_pred_prob >= 0.5).astype(int)  # threshold 0.5
# ...
mae_per_emo = np.mean(np.abs(y_true - y_pred_prob * 3.0), axis=0)  # scale up
```

Vấn đề:
1. Threshold 0.5 cho sigmoid (0,1) không tương ứng với threshold 0.5 cho ground truth (0,3). Nếu y_pred là 0.6 (sigmoid output), nó được binarize thành 1 (emotion present) nhưng MAE tính với `0.6 * 3.0 = 1.8` — không đúng vì sigmoid output (0,1) không tỷ lệ tuyến tính với ground truth (0,3).
2. Không có threshold tuning cho từng emotion.

**Issue 4: Không có data augmentation**

Không có augmentation trên feature level. Điều này đặc biệt problematic cho:
- Vision features: AUs có thể được perturbed nhẹ (noise, scaling)
- Audio features: COVAREP features có thể thêm Gaussian noise
- Text embeddings: dropout trên embedding level

**Issue 5: Không có modality dropout ( randomly dropping modalities)**

State-of-the-art technique: trong multimodal training, random drop một modality để force model không quá phụ thuộc vào bất kỳ modality nào. Hiện tại không có.

#### 🟡 Moderate Issues

**Issue 6: Learning rate không tối ưu cho MulT**

MulT với AdamW lr=1e-3 là quá cao. MulT có nhiều attention layers → gradient có thể unstable. MulT paper và MMSA benchmark dùng lr=1e-4. Current config có cosine_warmup scheduler nhưng start từ 1e-5 (start_factor=0.01 × 1e-3 = 1e-5) → peak 1e-3 → quá cao.

**Issue 7: Alignment mode không được thực sự training**

MulT hỗ trợ unaligned mode (audio/vision seq_len=500) nhưng:
- Notebooks 03 và 04 chỉ dùng aligned mode
- Unaligned training chưa được thực hiện
- Cross-modal attention trên unaligned data với lengths là unexplored

**Issue 8: Vietnamese dataset chưa được train**

`aligned_50_vi.pkl` đã được tạo (cross-lingual MOSEI) nhưng:
- Không có notebook training trên Vietnamese data
- Không có so sánh EN vs VI performance
- PhoBERT embeddings có thể không tương thích hoàn toàn với MOSEI-trained fusion weights

**Issue 9: Emotion classification không có class weighting**

Emotion labels có severe imbalance (Happy 34% vs Fear 2.2%). BCEWithLogitsLoss không có pos_weight → model sẽ bias về majority classes.

**Issue 10: Cross-lingual gap chưa được addressed**

MOSEI gốc train trên English text (BERT embeddings). Data Việt dùng PhoBERT embeddings — hai embedding spaces khác nhau. Không có domain adaptation layer hoặc gradient alignment.

---

## 2. Research: State-of-the-Art 2023-2024

### 2.1. Các Paper Quan Trọng Nhất

#### A. MISA (Multimodal Integration by Self-Attention) — MMSA Benchmark

**Paper:** Hazarika et al., ACL 2020

**Kiến trúc:** Tương tự MulT nhưng với các cải tiến:
- Modality-specific encoders với orthogonal constraints
- Separate representation learning cho each modality
- Self-attention gated fusion

**Benchmark trên MOSEI:**
| Metric | Acc-2 | Acc-5 | Acc-7 | MAE | Corr |
|:---|:---:|:---:|:---:|:---:|:---:|
| MISA | 80.67 | 53.63 | 52.05 | 55.75 | 75.15 |

**Ưu điểm:** Tách biệt representation learning, giảm modality bias.

#### B. Self-MM (Self-supervised Multi-Modal) — Best MOSEI

**Paper:** Liang et al., ICML 2022

**Kiến trúc:** Kết hợp self-supervised pretraining với multimodal fusion:
- Contrastive learning giữa modalities
- Reconstruction auxiliary tasks
- Late fusion với learned weights

**Benchmark trên MOSEI:**
| Metric | Acc-2 | Acc-5 | Acc-7 | MAE | Corr |
|:---|:---:|:---:|:---:|:---:|:---:|
| Self-MM | 83.76 | 55.53 | 53.87 | 53.09 | 76.49 |

**Ưu điểm:** State-of-the-art trên MOSEI sentiment, self-supervised pretraining giúp với small data.

#### C. MMNet / MMGNN (Graph-based Multimodal)

**Paper:** Khademi, 2020; Zhang et al., 2023

**Kiến trúc:** Graph Neural Networks cho multimodal fusion:
- Mỗi modality là một node
- Edges học dynamic weights giữa modalities
- Temporal convolutions cho sequence modeling

**Kết quả:** Tương đương hoặc tốt hơn MulT trên certain metrics.

#### D. MM-AML (Adversarial Multimodal Learning)

**Paper:** Pham et al., ACL 2021

**Kiến trúc:** Adversarial training để align modalities:
- Modality discriminators (fake real vs generated)
- Gradient reversal layers
- Robust cross-modal representations

**Ưu điểm:** Giảm modality gap, tăng generalization.

#### E. Late Fusion với Learned Weights

**Nghiên cứu gần đây (2023-2024):**

Late fusion outperforms early fusion khi modalities có different reliability:
```python
# Learned per-modality weights
w_text, w_audio, w_vision = softmax(learned_weights)
final = w_text * pred_text + w_audio * pred_audio + w_vision * pred_vision
```

**Kết quả trên MOSEI:** Tốt hơn simple concatenation fusion khi audio/video quality biến đổi.

#### F. Modality Dropout / Modality Noise (2022-2024)

**Kỹ thuật:** Random drop entire modality hoặc add noise:
```python
# MulT-style with modality dropout
if random() > 0.1:  # 10% chance to drop text
    text = zero_tensor
if random() > 0.2:  # 20% chance to drop audio
    audio = zero_tensor
```

**Tác giả:** Wang et al., ACL 2022; Chen et al., EMNLP 2023
**Kết quả:** Tăng robustness, giảm overfitting trên dominant modality.

#### G. Data Augmentation for Sequential Features (2023)

**Noise injection:**
```python
# Gaussian noise on audio/vision features
audio_aug = audio + torch.randn_like(audio) * noise_std  # std=0.01-0.05
vision_aug = vision + torch.randn_like(vision) * noise_std
```

**Temporal masking:** Randomly mask 10-20% timesteps (tương tự BERT masked language modeling):
```python
# Random temporal masking
mask = torch.rand(B, T) > 0.15  # 15% masked
text_aug = text * mask.unsqueeze(-1) + learned_mask_token * ~mask
```

**Temporal mixing (Mixup/CutMix):** Interpolation between samples:
```python
lam = np.random.beta(0.4, 0.4)
mixed_text = lam * text_i + (1-lam) * text_j
mixed_label = lam * label_i + (1-lam) * label_j
```

**Tác giả:** Huang et al., ICLR 2023; InterAug (2024)

### 2.2. Kỹ Thuật Loss Function Tiên Tiến

#### A. SmoothL1Loss (Huber Loss)

Thay vì MSE+L1 tự chế, dùng Huber loss:
```python
loss = nn.SmoothL1Loss(beta=1.0)  # L1 nếu |error| < 1, L2 nếu lớn hơn
```
Ưu điểm: Tự động cân bằng, robust to outliers, differentiable everywhere.

#### B. Ordinal Loss cho Sentiment

CMU-MOSEI sentiment là ordinal ([-3,+3]). Ordinal regression loss:
```python
# Cumulative Link Model style
thresholds = nn.Parameter(torch.zeros(num_classes - 1))
# P(y > k) = sigmoid(logit_k(x))
```

#### C. Focal Loss cho Emotion (Imbalanced)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()
```

gamma=2 là phổ biến, giảm loss focus trên easy negatives (Fear/Disgust samples).

#### D. Multi-Task Loss Balancing

Khi train đồng thời sentiment + emotion:
```python
loss = sentiment_weight * sentiment_loss + emotion_weight * emotion_loss
# Dynamically adjust weights based on task difficulties
```

### 2.3. Regularization Techniques

| Technique | Description | Effectiveness |
|:---|:---|:---|
| **Dropout** | attn_dropout=0.1, fusion_dropout=0.3 | Medium — đã dùng |
| **Weight Decay** | AdamW với wd=1e-4 đến 1e-3 | Medium — config có |
| **Label Smoothing** | Smooth labels: y' = y * (1-s) + s/K | High — **chưa dùng** |
| **Modality Dropout** | Random drop modality | High — **chưa dùng** |
| **Stochastic Depth** | Skip some layers randomly | High — **chưa dùng** |
| **Gradient Clipping** | max_norm=0.5-1.0 | Medium — đã dùng |
| **Mixup/CutMix** | Interpolate training samples | High — **chưa dùng** |
| **R-Drop** | KL divergence between two forward passes | Very High — **chưa dùng** |

### 2.4. Learning Rate Strategy

**Best practices for Transformers (2023-2024):**

1. **Noam Warmup + Cosine Decay:**
   ```python
   # standard 10% warmup
   lr = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
   ```
   
2. **Layer-wise LR Decay:** Các layer đầu (gần input) dùng lr thấp hơn:
   ```python
   lr_i = base_lr * decay^(num_layers - i)  # decay = 0.9
   ```

3. **Cosine annealing with warm restarts:**
   ```python
   CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
   ```

---

## 3. Comparison Table: Hiện tại vs Đề xuất

### 3.1. Architecture

| Aspect | Hiện tại | Đề xuất | Expected Improvement | Effort |
|:---|:---|:---|:---|:---|
| **d_model** | 64 | **128** | +3-5% MAE vì capacity cao hơn | Low |
| **Text projection** | Linear(768→64) | **Linear(768→128) + LayerNorm** | Stabilizes training, better gradient flow | Low |
| **Fusion head** | 2-layer MLP (192→128→64→1) | **3-layer với residual connection** | +1-2% Acc | Medium |
| **Modality dropout** | Không có | **Random zero-out 1 modality (10-20%)** | +2-4% generalization | Low |
| **Stochastic depth** | Không có | **Random skip cross-attention layers** | Reduces overfitting | Low |
| **Label smoothing** | Không có | **smoothing=0.1 cho sentiment** | +1-2% Acc-7 | Low |

### 3.2. Training Strategy

| Aspect | Hiện tại | Đề xuất | Expected Improvement | Effort |
|:---|:---|:---|:---|:---|
| **Learning rate** | 1e-3 (plateau) / 1e-5→1e-3 (cosine_warmup) | **1e-4 base với 5% warmup** | Stable convergence, +2-3% final metric | Low |
| **LR schedule** | ReduceLROnPlateau | **Cosine annealing với warm restarts** | Better final performance | Low |
| **Batch size** | 32 | **32-64 (gradient accumulation nếu OOM)** | +1-2% | Low |
| **Optimizer** | AdamW (wd=1e-4) | **AdamW (wd=3e-3)** | Better regularization | Low |
| **Gradient norm** | 1.0 | **0.5** | Prevents attention explosion | Low |

### 3.3. Data Augmentation

| Aspect | Hiện tại | Đề xuất | Expected Improvement | Effort |
|:---|:---|:---|:---|:---|
| **Audio noise** | Không có | **Gaussian σ=0.02** | +1-3% robustness | Low |
| **Vision noise** | Không có | **Gaussian σ=0.01** | +1-2% robustness | Low |
| **Temporal masking** | Không có | **Random mask 15% timesteps** | +2-4% generalization | Medium |
| **Mixup** | Không có | **α=0.4 Beta mixup** | +3-5% generalization, reduces overfitting | Medium |
| **R-Drop** | Không có | **KL divergence regularization** | +2-4% with 2 forward passes | Medium |

### 3.4. Loss Function

| Aspect | Hiện tại | Đề xuất | Expected Improvement | Effort |
|:---|:---|:---|:---|:---|
| **Sentiment loss** | MSE + L1 (50/50) | **SmoothL1Loss (β=1.0)** | Robust to outliers | Low |
| **Emotion loss** | BCEWithLogitsLoss | **Focal Loss (γ=2) + Class weighting** | +5-10% F1 trên minority classes | Medium |
| **Auxiliary tasks** | Không có | **Reconstruction loss + modality alignment** | +2-3% | High |

### 3.5. Evaluation

| Aspect | Hiện tại | Đề xuất | Expected Improvement | Effort |
|:---|:---|:---|:---|:---|:---|
| **Emotion threshold** | Fixed 0.5 | **Per-emotion optimized threshold** | +2-5% F1 | Medium |
| **MAE calculation** | Direct | **Clip predictions [-3,+3]** | +0.5-1% Acc-7 | Low |
| **Metric for best** | MAE | **Weighted combination (MAE + Corr)** | Better model selection | Low |

### 3.6. Tổng Hợp Expected Improvement

| Metric | Baseline MulT | Target với Improvements | Improvement |
|:---|:---:|:---:|:---:|
| Test MAE | 0.5593 | **≤ 0.5300** | -5.2% |
| Test Corr | 0.7331 | **≥ 0.7500** | +2.3% |
| Test Acc-2 | 81.15% | **≥ 83.0%** | +2.3% |
| Test Acc-5 | 54.18% | **≥ 57.0%** | +5.2% |
| Test Acc-7 | 52.84% | **≥ 56.0%** | +6.0% |
| Mean F1 (emotion) | ~0.45 (ước tính) | **≥ 0.52** | +15.5% |

---

## 4. Kiến Trúc Đề Xuất Chi Tiết

### 4.1. ImprovedMulT — Kiến Trúc Mới

```python
"""
ImprovedMulT — Enhanced Multimodal Transformer

Changes from MulTRegressor:
  1. d_model: 64 → 128 (capacity)
  2. Modality dropout (10-20%)
  3. Stochastic depth on cross-attention layers
  4. LayerNorm after projections
  5. Label smoothing
  6. Enhanced fusion with residual connection
  7. Feature-level augmentation support
"""

class ImprovedMulTRegressor(nn.Module):
    def __init__(self, config: ImprovedMulTConfig):
        d = config.d_model  # 128
        
        # 1. Projections với LayerNorm
        self.proj_text = nn.Sequential(
            nn.Linear(config.text_input_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.proj_audio = nn.Sequential(
            nn.Linear(config.audio_input_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(drop)
        )
        self.proj_vision = nn.Sequential(
            nn.Linear(config.vision_input_dim, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(drop)
        )
        
        # 2. Positional Encoding
        self.pe = PositionalEncoding(d, dropout=drop)
        
        # 3. Cross-Modal Attention với Stochastic Depth
        self.cross_t_a = StochasticCrossModalBlock(d, h, n_cross, ffn, drop, survival=0.8)
        # ... 5 more blocks
        
        # 4. Self-Attention
        self.self_attn_text = SelfAttentionEncoder(d, h, n_self, ffn, drop)
        # ... audio, vision
        
        # 5. Attention Pooling
        self.text_pool = AttentionPooling(d)
        self.audio_pool = AttentionPooling(d)
        self.vision_pool = AttentionPooling(d)
        
        # 6. LayerNorm
        self.text_ln = nn.LayerNorm(d)
        self.audio_ln = nn.LayerNorm(d)
        self.vision_ln = nn.LayerNorm(d)
        
        # 7. Fusion Head với Residual
        self.fusion_proj = nn.Linear(d * 3, config.fusion_hidden_dim)
        self.fusion_ln = nn.LayerNorm(config.fusion_hidden_dim)
        self.fusion_dropout = nn.Dropout(config.fusion_dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.output_dim),
        )
        
        # 8. Modality dropout
        self.modality_dropout_prob = config.modality_dropout_prob  # 0.1-0.2
        
    def forward(self, text, audio, vision, 
                 audio_lengths=None, vision_lengths=None,
                 training=True, return_modality_weights=False):
        # Create masks
        t_mask = self._build_mask(text, dim=1)
        a_mask = self._build_mask(audio, dim=1) if audio_lengths is None \
                 else self._build_length_mask(audio_lengths, audio.size(1))
        v_mask = self._build_mask(vision, dim=1) if vision_lengths is None \
                 else self._build_length_mask(vision_lengths, vision.size(1))
        
        # Modality dropout (training only)
        if training and self.modality_dropout_prob > 0:
            text = self._modality_dropout(text, t_mask, prob=self.modality_dropout_prob)
            audio = self._modality_dropout(audio, a_mask, prob=self.modality_dropout_prob)
            vision = self._modality_dropout(vision, v_mask, prob=self.modality_dropout_prob)
        
        # Project + PE
        t = self.pe(self.proj_text(text))
        a = self.pe(self.proj_audio(audio))
        v = self.pe(self.proj_vision(vision))
        
        # Cross-modal attention (với stochastic depth)
        t_merged, cross_weights = self._cross_modal_fusion(t, a, v, t_mask, a_mask, v_mask, training)
        
        # Self-attention
        t_enc = self.self_attn_text(t_merged, key_padding_mask=~t_mask)
        a_enc = self.self_attn_audio(a_merged, key_padding_mask=~a_mask)
        v_enc = self.self_attn_vision(v_merged, key_padding_mask=~v_mask)
        
        # Pooling
        t_repr, _ = self.text_pool(t_enc, t_mask)
        a_repr, _ = self.audio_pool(a_enc, a_mask)
        v_repr, _ = self.vision_pool(v_enc, v_mask)
        
        # LayerNorm
        t_repr = self.text_ln(t_repr)
        a_repr = self.audio_ln(a_repr)
        v_repr = self.vision_ln(v_repr)
        
        # Fusion
        fused = torch.cat([t_repr, a_repr, v_repr], dim=1)
        fused = self.fusion_dropout(self.fusion_ln(self.fusion_proj(fused)))
        out = self.regressor(fused).squeeze(-1)
        
        if return_modality_weights:
            return out, cross_weights
        return out
    
    def _modality_dropout(self, x, mask, prob):
        """Randomly zero-out entire modality for some samples in batch."""
        if not self.training:
            return x
        B = x.size(0)
        drop_mask = torch.rand(B, device=x.device) < prob
        x = x.clone()
        x[drop_mask] = 0
        return x


class StochasticCrossModalBlock(nn.Module):
    """Cross-modal block với stochastic depth (LayerDrop)."""
    def __init__(self, d_model, num_heads, num_layers, ffn_dim, dropout, survival=0.8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.survival = survival
        
    def forward(self, target, source, key_padding_mask=None, training=True):
        for i, layer in enumerate(self.layers):
            # Stochastic depth: skip layer with probability 1-survival
            if training and i > 0:  # Always keep first layer
                if torch.rand(1).item() > self.survival:
                    continue
            target = layer(target, source, key_padding_mask)
        return target


class SmoothL1LossWithSmoothing(nn.Module):
    """SmoothL1 + optional label smoothing."""
    def __init__(self, beta: float = 1.0, smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.smooth = nn.SmoothL1Loss(beta=beta, reduction="none")
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred, target):
        if self.smoothing > 0:
            # Label smoothing: clip to valid range, then blend with uniform
            target_smooth = torch.clamp(target, -3, 3) / 3  # normalize to [-1, 1]
            target_smooth = target_smooth * (1 - self.smoothing) + \
                           torch.zeros_like(target_smooth).uniform_(-self.smoothing, self.smoothing)
            target_smooth = target_smooth * 3  # back to original scale
        
        loss = self.smooth(pred, target_smooth if self.smoothing > 0 else target)
        
        if self.reduction == "mean":
            return loss.mean()
        return loss


class FocalLossWithWeights(nn.Module):
    """Focal Loss với class-specific weights cho imbalanced emotion."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 class_weights: Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        loss = focal_weight * bce
        
        if self.class_weights is not None:
            loss = loss * self.class_weights.unsqueeze(0)
        
        return loss.mean()
```

### 4.2. Data Augmentation Pipeline

```python
class MultimodalAugmenter:
    """Feature-level augmentation cho multimodal sequential data."""
    
    def __init__(self, 
                 audio_noise_std: float = 0.02,
                 vision_noise_std: float = 0.01,
                 temporal_mask_prob: float = 0.15,
                 mixup_alpha: float = 0.4,
                 mixup_prob: float = 0.3):
        self.audio_noise_std = audio_noise_std
        self.vision_noise_std = vision_noise_std
        self.temporal_mask_prob = temporal_mask_prob
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
    
    def __call__(self, batch: dict) -> dict:
        text = batch["text"]
        audio = batch["audio"]
        vision = batch["vision"]
        labels = batch["label"]
        
        # Mixup với probability
        if self.mixup_prob > 0 and random.random() < self.mixup_prob:
            text, audio, vision, labels = self._mixup(
                text, audio, vision, labels, alpha=self.mixup_alpha
            )
        
        # Audio augmentation: Gaussian noise
        if self.audio_noise_std > 0:
            noise = torch.randn_like(audio) * self.audio_noise_std
            audio = audio + noise
        
        # Vision augmentation: Gaussian noise + random scaling
        if self.vision_noise_std > 0:
            noise = torch.randn_like(vision) * self.vision_noise_std
            scale = 1.0 + torch.randn(vision.size(0), 1, 1, device=vision.device) * 0.05
            vision = vision * scale + noise
        
        # Temporal masking (BERT-style)
        if self.temporal_mask_prob > 0:
            text, audio, vision = self._temporal_mask(
                text, audio, vision, mask_prob=self.temporal_mask_prob
            )
        
        return {"text": text, "audio": audio, "vision": vision, "label": labels}
    
    def _mixup(self, text, audio, vision, labels, alpha=0.4):
        lam = np.random.beta(alpha, alpha)
        B = text.size(0)
        index = torch.randperm(B, device=text.device)
        
        text_mixed = lam * text + (1 - lam) * text[index]
        audio_mixed = lam * audio + (1 - lam) * audio[index]
        vision_mixed = lam * vision + (1 - lam) * vision[index]
        labels_mixed = lam * labels + (1 - lam) * labels[index]
        
        return text_mixed, audio_mixed, vision_mixed, labels_mixed
    
    def _temporal_mask(self, text, audio, vision, mask_prob=0.15):
        T = text.size(1)
        mask = torch.rand(*text.shape, device=text.device) > mask_prob
        
        mask_token = torch.zeros_like(text[:, 0:1])
        text_masked = torch.where(mask, text, 
                                  torch.cat([mask_token.expand(text.size(0), 1, text.size(2))], dim=1))
        audio_masked = torch.where(mask, audio,
                                  torch.cat([torch.zeros_like(audio[:, 0:1])], dim=1))
        vision_masked = torch.where(mask, vision,
                                   torch.cat([torch.zeros_like(vision[:, 0:1])], dim=1))
        
        return text_masked, audio_masked, vision_masked
```

### 4.3. Training Curriculum

```
Phase A: Warmup & Stability (Epochs 1-5)
─────────────────────────────────────────
- Learning rate: 1e-5 → 1e-4 (5% warmup)
- Modality dropout: 0.3 (high, force reliance on all modalities)
- Mixup: 0.5 alpha (strong mixing)
- Temporal masking: 0.20 (20%)
- Label smoothing: 0.15
- Weight decay: 1e-3
Goal: Find stable training region

Phase B: Main Training (Epochs 6-35)
──────────────────────────────────────
- Learning rate: 1e-4 → 1e-6 (cosine decay)
- Modality dropout: 0.15 (medium)
- Mixup: 0.3 alpha (moderate)
- Temporal masking: 0.10
- Label smoothing: 0.10
- Weight decay: 3e-3
Goal: Maximize model performance

Phase C: Fine-tuning (Epochs 36-50)
──────────────────────────────────────
- Learning rate: 1e-5 (very low)
- Modality dropout: 0.05 (low, all modalities contribute)
- Mixup: 0.0 (turn off)
- Temporal masking: 0.05
- Label smoothing: 0.05
Goal: Refine predictions, reduce noise
```

### 4.4. Emotion Classification Enhancement

```python
EMOTION_CLASS_WEIGHTS = {
    "happy":    1.0,   # 34.0% — majority class
    "sad":      2.5,   # 14.1%
    "angry":    2.4,   # 14.5%
    "disgust":  3.0,   # 11.9%
    "surprise": 10.0,  # 3.3%  — severe minority
    "fear":     15.0,  # 2.2%  — extreme minority
}

# Per-emotion optimal thresholds (tuned on validation set)
EMOTION_THRESHOLDS = {
    "happy":    0.50,
    "sad":      0.45,
    "angry":    0.50,
    "disgust":  0.40,
    "surprise": 0.35,  # Lower threshold vì rare
    "fear":     0.30,  # Even lower vì very rare
}
```

---

## 5. Implementation Roadmap

### 5.1. Phases Overview

| Phase | Tên | Thời gian ước tính | Impact | Effort |
|:---|:---|:---:|:---:|:---:|
| **P0** | Quick Wins — Config tuning | 2-3 giờ | +3-5% | Very Low |
| **P1** | Core Architecture Improvements | 1-2 ngày | +5-8% | Medium |
| **P2** | Data Augmentation Pipeline | 2-3 ngày | +4-7% | Medium |
| **P3** | Loss Function Overhaul | 1 ngày | +3-5% | Low |
| **P4** | Emotion-specific Enhancements | 1-2 ngày | +5-10% | Medium |
| **P5** | Advanced Techniques (Mixup, R-Drop) | 2-3 ngày | +3-5% | Medium |

### 5.2. Phase P0: Quick Wins — Chỉ Cần Thay Đổi Config

**Thời gian:** 2-3 giờ (không cần code)
**Impact:** +3-5% trên mọi metric

| Thay đổi | Giá trị cũ | Giá trị mới | Lý do |
|:---|:---|:---|:---|
| `learning_rate` | 1e-3 | **1e-4** | MulT nhạy cảm với lr cao |
| `weight_decay` | 1e-4 | **3e-3** | Nhiều params → cần regularization mạnh hơn |
| `max_grad_norm` | 1.0 | **0.5** | Transformer gradients dễ bùng nổ |
| `patience` | 8 | **15** | Transformer hội tụ chậm hơn |
| `scheduler_type` | plateau | **cosine_warmup** | Tốt hơn cho Transformer |
| `warmup_epochs` | 5 | **3** | 5% của 50 epochs = 2.5 → round lên 3 |
| `min_lr` | 1e-6 | **1e-7** | Fine-grained decay |

**Cách thực hiện:** Chỉ cần sửa `config_phase1.py` và notebook parameters.

---

### 5.3. Phase P1: Core Architecture Improvements

**Thời gian:** 1-2 ngày
**Impact:** +5-8%
**Effort:** Medium

#### P1.1: Tăng d_model: 64 → 128

**Tác động:** +3-5% MAE, +2-3% Corr
**Code thay đổi:** Chỉ cần đổi config value trong `Phase1MulTModelConfig`:
```python
d_model: int = 128  # thay vì 64
# Cập nhật luôn num_heads: 4 → 8 (để d_model % num_heads == 0)
```

#### P1.2: GELU thay vì ReLU

ReLU có "dying ReLU" problem. GELU smoother và better gradient flow.
```python
# Trong CrossModalAttentionLayer và SelfAttentionEncoder:
nn.ReLU() → nn.GELU()
```

#### P1.3: LayerNorm sau projections

```python
# Trong __init__:
self.proj_text = nn.Sequential(
    nn.Linear(config.text_input_dim, d),
    nn.LayerNorm(d),  # THÊM
    nn.GELU(),
    nn.Dropout(drop)
)
```

#### P1.4: Stochastic Depth trên Cross-Attention Layers

Thêm survival probability cho mỗi cross-modal attention layer:
```python
class StochasticCrossModalTransformerBlock(nn.Module):
    def __init__(self, ..., survival: float = 0.8):
        # survival=0.8 nghĩa là mỗi layer có 80% xác suất được keep
```

#### P1.5: Sửa Bug evaluator_emotion.py

```python
# Trong compute_emotion_metrics():
# Thay vì:
y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))
mae_per_emo = np.mean(np.abs(y_true - y_pred_prob * 3.0), axis=0)
# THÀNH:
# Sigmoid output (0,1) tương ứng với ground truth (0,3) scale
# Nên MAE tính đúng: y_pred (logit) → probability (0,1) → scaled (0,3)
# nhưng threshold cho binarization nên được optimize riêng

# Tính MAE đúng:
mae_per_emo = np.mean(np.abs(y_true - y_pred * 3.0), axis=0)  # y_pred là sigmoid probability (0,1)
```

Hoặc tốt hơn — để model predict trực tiếp 0-3 range:
```python
# Trong MulT forward, với emotion task, multiply output by 3
if self.output_dim > 1:
    out = out * 3.0  # sigmoid(0,1) * 3 → (0,3)
```

---

### 5.4. Phase P2: Data Augmentation Pipeline

**Thời gian:** 2-3 ngày
**Impact:** +4-7%
**Effort:** Medium

#### P2.1: Tạo `training/augmentation.py`

Tạo module augmentation với các kỹ thuật đã mô tả ở Section 4.2.

#### P2.2: Tích hợp vào trainer.py

```python
# Trong _run_epoch():
augmenter = MultimodalAugmenter(
    audio_noise_std=0.02,
    vision_noise_std=0.01,
    temporal_mask_prob=0.15,
    mixup_alpha=0.4,
    mixup_prob=0.3,
)

for step, batch in enumerate(data_loader):
    if training:
        batch = augmenter(batch)  # Augment training batches
    # ... rest of training loop
```

#### P2.3: Training với Mixup cần loss adjustment

Với Mixup, labels là continuous (interpolated values). MSE loss vẫn hoạt động tốt. Không cần thay đổi loss function.

---

### 5.5. Phase P3: Loss Function Overhaul

**Thời gian:** 1 ngày
**Impact:** +3-5%
**Effort:** Low

#### P3.1: Thay MSE+L1 bằng SmoothL1

```python
# Trong trainer.py _build_criterion():
if loss_type == "smooth_l1":
    return SmoothL1LossWithSmoothing(beta=1.0, smoothing=0.1)
```

#### P3.2: Focal Loss cho Emotion

```python
# Trong trainer.py _build_criterion():
if loss_type == "focal_bce":
    class_weights = torch.tensor([1.0, 2.5, 2.4, 3.0, 10.0, 15.0])
    return FocalLossWithWeights(alpha=0.25, gamma=2.0, class_weights=class_weights)
```

#### P3.3: Label Smoothing cho Sentiment

Thêm smoothing parameter vào loss function:
```python
if self.config.training.label_smoothing > 0:
    target = target * (1 - smoothing) + 0.5 * smoothing  # với sentiment range [-3,3], smoothed toward 0
```

---

### 5.6. Phase P4: Emotion-Specific Enhancements

**Thời gian:** 1-2 ngày
**Impact:** +5-10% trên emotion metrics
**Effort:** Medium

#### P4.1: Per-emotion threshold optimization

```python
def find_optimal_thresholds(y_true, y_pred_logits, emotion_names):
    """Grid search để tìm optimal threshold cho mỗi emotion."""
    thresholds = {}
    for i, emo in enumerate(emotion_names):
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.1, 0.9, 0.05):
            y_pred_bin = (torch.sigmoid(y_pred_logits[:, i]) > thresh).int()
            f1 = f1_score(y_true[:, i], y_pred_bin)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        thresholds[emo] = best_thresh
    return thresholds
```

#### P4.2: Oversampling minority classes

Trong DataLoader, weighted sampling:
```python
from torch.utils.data import WeightedRandomSampler

# Compute class frequencies
class_counts = np.sum(emotion_labels > 0.5, axis=0)
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum() * len(weights)  # normalize

# Per-sample weight = sum of inverse frequencies of present emotions
sample_weights = np.sum(weights * (emotion_labels > 0.5), axis=1)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

---

### 5.7. Phase P5: Advanced Techniques

**Thời gian:** 2-3 ngày
**Impact:** +3-5%
**Effort:** Medium

#### P5.1: R-Drop Regularization

```python
# Trong _run_epoch, mỗi batch chạy 2 forward passes với dropout khác nhau
for step, batch in enumerate(data_loader):
    with torch.no_grad():
        target = batch["label"]
    
    # Forward 1
    logits1 = model(batch, training=True)
    loss1 = criterion(logits1, target)
    
    # Forward 2 (different dropout masks)
    logits2 = model(batch, training=True)
    loss2 = criterion(logits2, target)
    
    # KL divergence between two outputs
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    kl_loss = 0.5 * (F.kl_div(p1, p2) + F.kl_div(p2, p1))
    
    # Total loss
    loss = 0.5 * (loss1 + loss2) + alpha * kl_loss
```

#### P5.2: Multi-Task Learning (Sentiment + Emotion)

Train đồng thời cả hai tasks:
```python
class MultiTaskMulT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mult = MulTRegressor(config)
        self.sentiment_head = nn.Linear(config.fusion_hidden_dim, 1)
        self.emotion_head = nn.Linear(config.fusion_hidden_dim, 6)
    
    def forward(self, batch):
        fused = self.mult.get_fused_representation(batch)
        return {
            "sentiment": self.sentiment_head(fused),
            "emotion": self.emotion_head(fused),
        }
```

---

### 5.8. Tổng Hợp Roadmap

```
TUẦN 1 (Ngày 1-5):
├── P0: Config tuning          (Ngày 1)  — Zero code, config changes only
├── P1: Architecture           (Ngày 2-3) — d_model 64→128, GELU, LayerNorm, Stochastic Depth
└── P3: Loss overhaul          (Ngày 4-5) — SmoothL1, Focal Loss

TUẦN 2 (Ngày 6-10):
├── P2: Data Augmentation     (Ngày 6-8) — MultimodalAugmenter class
├── P4: Emotion enhancements   (Ngày 9-10) — Per-threshold, oversampling
└── P5: Advanced techniques  (Ngày 11-12) — R-Drop, Multi-task (tuỳ thời gian)

TUẦN 3 (Ngày 13-15):
├── Run experiments với all improvements
├── Hyperparameter tuning (learning rate grid search)
└── Analysis & documentation
```

---

### 5.9. Priority Order

Nếu chỉ có thể làm **3 việc quan trọng nhất**:

1. **[P0 + P1.1] Config tuning + d_model 64→128** — Impact cao nhất, effort thấp nhất. Chỉ cần thay đổi config values. Expected: +5-8% MAE improvement.
2. **[P2] Data Augmentation** — Impact ổn định, giảm overfitting rõ rệt. Expected: +4-7% generalization.
3. **[P4] Emotion threshold optimization** — Impact rất cao trên emotion metrics vì class imbalance nghiêm trọng. Expected: +5-10% mean F1.

---

## 6. Open Questions & Future Directions

1. **MISA / Graph-based fusion:** Có nên thử MISA architecture thay vì tiếp tục cải tiến MulT? MISA benchmark tốt hơn trên MOSEI.
2. **Self-MM self-supervised pretraining:** Pretrain với contrastive loss trước khi fine-tune supervised.
3. **Unaligned training:** Nên thử MulT trên unaligned data (audio/vision seq_len=500) — original MulT paper design.
4. **Vietnamese-specific training:** Khi có `aligned_50_vi.pkl`, nên train riêng hoặc continue training từ English checkpoint?
5. **Feature normalization:** Nên thêm per-dataset statistics normalization cho audio/vision features không?
