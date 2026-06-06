# 🏗️ Tài Liệu Triển Khai Kiến Trúc MulT (Multimodal Transformer)

## Dành cho Agent Code — Hướng dẫn triển khai chi tiết

> **Mục tiêu:** Chuẩn bị, sửa lỗi, tối ưu hóa và huấn luyện mô hình MulT trên CMU-MOSEI aligned features,
> đạt hoặc vượt benchmark MMSA (Test MAE ≤ 0.5593, Test Corr ≥ 0.7331).

---

## Mục Lục

1. [Bối Cảnh & Động Lực](#1-bối-cảnh--động-lực)
2. [Kiến Trúc MulT — Giải Thích Chi Tiết](#2-kiến-trúc-mult--giải-thích-chi-tiết)
3. [Kiểm Tra Mã Nguồn Hiện Tại (Code Audit)](#3-kiểm-tra-mã-nguồn-hiện-tại-code-audit)
4. [Danh Sách Bug & Vấn Đề Cần Sửa](#4-danh-sách-bug--vấn-đề-cần-sửa)
5. [Các Bước Triển Khai (Step-by-Step)](#5-các-bước-triển-khai-step-by-step)
6. [Cấu Hình Siêu Tham Số Đề Xuất](#6-cấu-hình-siêu-tham-số-đề-xuất)
7. [Tạo Notebook Huấn Luyện Colab](#7-tạo-notebook-huấn-luyện-colab)
8. [Kế Hoạch Xác Minh & Đánh Giá](#8-kế-hoạch-xác-minh--đánh-giá)
9. [Tham Khảo & Benchmark](#9-tham-khảo--benchmark)

---

## 1. Bối Cảnh & Động Lực

### 1.1. Tại sao chuyển sang MulT?

Qua Phase 1, hai mô hình dựa trên LSTM đã được huấn luyện và đánh giá:

| Mô hình | Test MAE | Test Corr | Test Acc-2 | Vấn đề chính |
|:---|:---:|:---:|:---:|:---|
| Baseline LSTM | 0.6071 | 0.6995 | 81.03% | Overfitting nghiêm trọng (~13x) |
| Improved LSTM | 0.5859 | 0.7229 | 81.37% | Overfitting vẫn cao |
| **Benchmark MulT (MMSA)** | **0.5593** | **0.7331** | **81.15%** | — Mục tiêu cần đạt |

**Kết luận:** Kiến trúc LSTM đã đạt trần (ceiling). MulT — Multimodal Transformer — sử dụng Cross-Modal Attention cho phép mỗi phương thức "nhìn vào" các phương thức khác, từ đó học được tương tác sâu hơn giữa Text, Audio, và Vision.

### 1.2. Paper gốc

> **"Multimodal Transformer for Unaligned Multimodal Language Sequences"**
> Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency
> ACL 2019 — [arXiv:1906.00295](https://arxiv.org/abs/1906.00295)

### 1.3. Dữ liệu sử dụng

- **File:** `data/MSA-Dataset/aligned_50.pkl`
- **Tổng mẫu:** 22,856 (Train: 16,326 | Valid: 1,871 | Test: 4,659)
- **Input dims:** Text (50, 768) | Audio (50, 74) | Vision (50, 35)
- **Label:** `regression_labels` — giá trị liên tục [-3.0, +3.0]

---

## 2. Kiến Trúc MulT — Giải Thích Chi Tiết

### 2.1. Tổng Quan Luồng Dữ Liệu

```
Input 3 phương thức
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 1: Projection — Chiếu về không gian chung     │
│                                                       │
│  Text  (B,50,768) ──► Linear(768→d) ──► (B,50,d)   │
│  Audio (B,50,74)  ──► Linear(74→d)  ──► (B,50,d)   │
│  Vision(B,50,35)  ──► Linear(35→d)  ──► (B,50,d)   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 2: Positional Encoding (sinusoidal)            │
│  Thêm thông tin vị trí cho mỗi timestep             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 3: Cross-Modal Attention (6 luồng)             │
│                                                       │
│  Text  ← Audio  (T hỏi A)    Audio ← Text  (A hỏi T)│
│  Text  ← Vision (T hỏi V)    Audio ← Vision(A hỏi V)│
│  Vision← Text   (V hỏi T)    Vision← Audio (V hỏi A)│
│                                                       │
│  Mỗi luồng = Stack N layers CrossModalAttention      │
│  Layer = MultiHeadAttn(Q=target, KV=source)          │
│        + LayerNorm + FFN + LayerNorm                 │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 4: Merge (Residual Sum)                        │
│                                                       │
│  T_merged = T_orig + T←A + T←V                      │
│  A_merged = A_orig + A←T + A←V                      │
│  V_merged = V_orig + V←T + V←A                      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 5: Self-Attention Transformer Encoder          │
│                                                       │
│  T_encoded = TransformerEncoder(T_merged)            │
│  A_encoded = TransformerEncoder(A_merged)            │
│  V_encoded = TransformerEncoder(V_merged)            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 6: Temporal Pooling → Representation           │
│  (Hiện tại: lấy timestep cuối [:, -1, :])           │
│  (Cải tiến: Attention Pooling hoặc Mean Pooling)     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  BƯỚC 7: Fusion Head                                 │
│  Concat [T_repr, A_repr, V_repr] → (B, 3*d)        │
│  → Linear(3*d → hidden) → ReLU → Dropout            │
│  → Linear(hidden → 1) → Sentiment Score             │
└─────────────────────────────────────────────────────┘
```

### 2.2. Cross-Modal Attention — Trái Tim Của MulT

```
         Target Modality              Source Modality
         (ví dụ: Text)               (ví dụ: Audio)
              │                            │
              ▼                            ▼
         ┌────────┐                   ┌────────┐
         │ Q = Wq │                   │ K = Wk │
         │        │                   │ V = Wv │
         └───┬────┘                   └───┬────┘
             │                            │
             ▼                            ▼
        ┌─────────────────────────────────────┐
        │    Scaled Dot-Product Attention      │
        │    Attn(Q,K,V) = softmax(QK^T/√d)V  │
        └────────────────┬────────────────────┘
                         │
                         ▼
                  Residual + LayerNorm
                         │
                         ▼
                   Feed-Forward Network
                         │
                         ▼
                  Residual + LayerNorm
                         │
                         ▼
                 Output (enriched target)
```

**Ý nghĩa:** Khi Text làm Query và Audio làm Key/Value, mô hình cho phép mỗi vị trí trong chuỗi Text "tìm kiếm" thông tin liên quan từ chuỗi Audio. Điều này cho phép Text "nghe" được ngữ điệu, giọng nói tương ứng.

### 2.3. So sánh kiến trúc với LSTM

| Khía cạnh | LSTM (Baseline/Improved) | MulT (Transformer) |
|:---|:---|:---|
| Encoder | BiLSTM xử lý tuần tự | Self-Attention xử lý song song |
| Cross-modal | Không có (chỉ concat sau encode) | 6 luồng Cross-Modal Attention |
| Temporal context | Hidden state tích lũy (quên dần) | Attention toàn cục mọi vị trí |
| Fusion | Concatenation / Gated Fusion | Residual Sum + Self-Attention |
| Xử lý padding | Bị ảnh hưởng (padding zeros) | Có thể mask (nếu triển khai) |
| Số tham số | ~1–2 triệu | ~2–5 triệu (tùy config) |

---

## 3. Kiểm Tra Mã Nguồn Hiện Tại (Code Audit)

### 3.1. Các file liên quan đến MulT

| File | Đường dẫn | Trạng thái | Ghi chú |
|:---|:---|:---:|:---|
| **Mô hình MulT** | `training/models/mult.py` | ✅ Có | 260 dòng, đầy đủ kiến trúc |
| **Config MulT** | `training/config_phase1.py` (class `Phase1MulTModelConfig`) | ✅ Có | Dòng 73–92 |
| **Main entrypoint** | `training/main_phase1.py` | ✅ Có | Hỗ trợ `model_type="mult"` (dòng 57–59) |
| **Trainer** | `training/trainer.py` | ✅ Có | Model-agnostic, dùng chung |
| **Dataset** | `training/dataset_mosei.py` | ⚠️ Bug | Validate shapes dùng `config.model.*` |
| **Models __init__** | `training/models/__init__.py` | ⚠️ Thiếu | Chưa export `MulTRegressor` |
| **Notebook Colab** | `notebooks/03_mult_*.ipynb` | ❌ Chưa có | Cần tạo mới |

### 3.2. Phân tích `mult.py` — Kiến trúc hiện tại

```python
# File: training/models/mult.py

class PositionalEncoding(nn.Module):     # Dòng 22-46  ✅ OK
class CrossModalAttentionLayer(nn.Module): # Dòng 53-96  ✅ OK
class CrossModalTransformerBlock(nn.Module): # Dòng 103-123  ✅ OK
class SelfAttentionEncoder(nn.Module):   # Dòng 130-153  ✅ OK
class MulTRegressor(nn.Module):          # Dòng 160-259  ⚠️ Cần cải tiến
```

### 3.3. Phân tích `Phase1MulTModelConfig` — Config hiện tại

```python
@dataclass
class Phase1MulTModelConfig:
    text_input_dim: int = 768     # BERT embedding dimension
    audio_input_dim: int = 74     # COVAREP features
    vision_input_dim: int = 35    # FACET Action Units

    d_model: int = 64             # ⚠️ Nhỏ — paper gốc dùng 40 nhưng trên MOSI
    num_heads: int = 4            # ⚠️ Cần d_model % num_heads == 0
    num_cross_layers: int = 3     # Số layers Cross-Modal Attention
    num_self_layers: int = 2      # Số layers Self-Attention
    ffn_dim: int = 128            # Feed-Forward hidden dim
    attn_dropout: float = 0.1     # Attention dropout

    fusion_hidden_dim: int = 128  # Fusion MLP hidden dim
    fusion_dropout: float = 0.3   # Fusion dropout
    output_dim: int = 1           # Regression output
```

---

## 4. Danh Sách Bug & Vấn Đề Cần Sửa

### 🔴 Bug 1: `dataset_mosei.py` — Validate shapes sai khi dùng MulT

**File:** `training/dataset_mosei.py`, dòng 58–66

**Vấn đề:** Hàm `_validate_shapes()` luôn tham chiếu `self.config.model.*` (là `Phase1ModelConfig` cho LSTM), KHÔNG phải `self.config.mult_model.*` (là `Phase1MulTModelConfig` cho MulT). Hiện tại vô tình hoạt động vì cả hai config đều có cùng giá trị mặc định `text_input_dim=768, audio_input_dim=74, vision_input_dim=35`, nhưng nếu ai đó thay đổi giá trị trong `Phase1MulTModelConfig`, validation sẽ sai.

```python
# BUG: Luôn dùng config.model (LSTM config) thay vì kiểm tra model_type
def _validate_shapes(self) -> None:
    n = len(self.labels)
    expected_seq = self.config.data.sequence_length
    if self.text.shape != (n, expected_seq, self.config.model.text_input_dim):  # ← BUG
        raise ValueError(f"Unexpected text shape: {self.text.shape}")
```

**Cách sửa:**
```python
def _validate_shapes(self) -> None:
    n = len(self.labels)
    expected_seq = self.config.data.sequence_length
    # Chọn config input dims dựa trên model_type
    if self.config.model_type == "mult":
        text_dim = self.config.mult_model.text_input_dim
        audio_dim = self.config.mult_model.audio_input_dim
        vision_dim = self.config.mult_model.vision_input_dim
    else:
        text_dim = self.config.model.text_input_dim
        audio_dim = self.config.model.audio_input_dim
        vision_dim = self.config.model.vision_input_dim

    if self.text.shape != (n, expected_seq, text_dim):
        raise ValueError(f"Unexpected text shape: {self.text.shape}")
    if self.audio.shape != (n, expected_seq, audio_dim):
        raise ValueError(f"Unexpected audio shape: {self.audio.shape}")
    if self.vision.shape != (n, expected_seq, vision_dim):
        raise ValueError(f"Unexpected vision shape: {self.vision.shape}")
```

---

### 🔴 Bug 2: `models/__init__.py` — Chưa export MulTRegressor

**File:** `training/models/__init__.py`

**Hiện tại:**
```python
from .unimodal_encoder import BiLSTMEncoder
from .early_fusion import EarlyFusionLSTMRegressor
from .improved_lstm import ImprovedLSTMRegressor
# ← Thiếu MulTRegressor
```

**Cách sửa — thêm dòng:**
```python
from .unimodal_encoder import BiLSTMEncoder
from .early_fusion import EarlyFusionLSTMRegressor
from .improved_lstm import ImprovedLSTMRegressor
from .mult import MulTRegressor  # ← THÊM
```

---

### 🟡 Vấn đề 3: `MulTRegressor.forward()` — Lấy timestep cuối (padding issue)

**File:** `training/models/mult.py`, dòng 252–255

**Vấn đề:** Giống Baseline LSTM, mô hình lấy `[:, -1, :]` (timestep cuối) làm representation. Với dữ liệu CMU-MOSEI aligned (đã pad về 50 timesteps), timestep cuối thường là padding zeros → representation bị nhiễu.

```python
# HIỆN TẠI: Lấy timestep cuối (bị ảnh hưởng bởi padding)
t_repr = t_encoded[:, -1, :]   # (B, d)
a_repr = a_encoded[:, -1, :]   # (B, d)
v_repr = v_encoded[:, -1, :]   # (B, d)
```

**Cách sửa — Thêm Attention Pooling (đã chứng minh hiệu quả trong Improved LSTM):**
```python
# TRONG __init__:
self.text_pool = AttentionPooling(d)
self.audio_pool = AttentionPooling(d)
self.vision_pool = AttentionPooling(d)

# TRONG forward:
# Tạo mask cho padding positions
t_mask = (text.abs().sum(dim=-1) > 1e-6)    # (B, S)
a_mask = (audio.abs().sum(dim=-1) > 1e-6)   # (B, S)
v_mask = (vision.abs().sum(dim=-1) > 1e-6)  # (B, S)

t_repr, _ = self.text_pool(t_encoded, t_mask)    # (B, d)
a_repr, _ = self.audio_pool(a_encoded, a_mask)   # (B, d)
v_repr, _ = self.vision_pool(v_encoded, v_mask)  # (B, d)
```

> **Lưu ý:** Class `AttentionPooling` đã tồn tại trong `training/models/improved_lstm.py` (dòng 8–28). Nên di chuyển nó sang file riêng hoặc import từ đó.

---

### 🟡 Vấn đề 4: Cross-Modal Attention — Thiếu Padding Mask

**File:** `training/models/mult.py`, dòng 83–96

**Vấn đề:** `nn.MultiheadAttention` hỗ trợ `key_padding_mask` nhưng hiện chưa được sử dụng. Khi source modality có padding zeros, attention sẽ "nhìn" vào noise.

```python
# HIỆN TẠI: Không dùng mask
attn_out, _ = self.cross_attn(query=target, key=source, value=source)

# CẢI TIẾN: Truyền padding mask
attn_out, _ = self.cross_attn(
    query=target, key=source, value=source,
    key_padding_mask=source_mask  # True = ignored position
)
```

> **Lưu ý:** `nn.MultiheadAttention` kỳ vọng `key_padding_mask` với `True = ignored` (ngược với convention trong `AttentionPooling` dùng `True = valid`). Cần đảo mask: `key_padding_mask = ~source_valid_mask`.

---

### 🟡 Vấn đề 5: Fusion Head quá đơn giản

**Hiện tại:** 1 hidden layer + ReLU + Dropout → Output

**Đề xuất cải tiến — tương tự Improved LSTM:**
```python
self.regressor = nn.Sequential(
    nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
    nn.LayerNorm(config.fusion_hidden_dim),  # Thêm LayerNorm
    nn.ReLU(),
    nn.Dropout(config.fusion_dropout),
    nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),  # Thêm 1 layer
    nn.ReLU(),
    nn.Dropout(config.fusion_dropout * 0.5),
    nn.Linear(config.fusion_hidden_dim // 2, config.output_dim),
)
```

---

## 5. Các Bước Triển Khai (Step-by-Step)

### Bước 1: Sửa Bug — `models/__init__.py` [5 phút]

**File:** `training/models/__init__.py`

```diff
 from .unimodal_encoder import BiLSTMEncoder
 from .early_fusion import EarlyFusionLSTMRegressor
 from .improved_lstm import ImprovedLSTMRegressor
+from .mult import MulTRegressor
```

---

### Bước 2: Sửa Bug — `dataset_mosei.py` [10 phút]

**File:** `training/dataset_mosei.py`

Thay thế method `_validate_shapes` (dòng 58–66) để chọn input dims dựa trên `config.model_type`:

```python
def _validate_shapes(self) -> None:
    n = len(self.labels)
    expected_seq = self.config.data.sequence_length

    # Chọn input dims dựa trên model_type
    if self.config.model_type == "mult":
        text_dim = self.config.mult_model.text_input_dim
        audio_dim = self.config.mult_model.audio_input_dim
        vision_dim = self.config.mult_model.vision_input_dim
    else:
        text_dim = self.config.model.text_input_dim
        audio_dim = self.config.model.audio_input_dim
        vision_dim = self.config.model.vision_input_dim

    if self.text.shape != (n, expected_seq, text_dim):
        raise ValueError(f"Unexpected text shape: {self.text.shape}, expected (n, {expected_seq}, {text_dim})")
    if self.audio.shape != (n, expected_seq, audio_dim):
        raise ValueError(f"Unexpected audio shape: {self.audio.shape}, expected (n, {expected_seq}, {audio_dim})")
    if self.vision.shape != (n, expected_seq, vision_dim):
        raise ValueError(f"Unexpected vision shape: {self.vision.shape}, expected (n, {expected_seq}, {vision_dim})")
```

---

### Bước 3: Tách `AttentionPooling` thành module dùng chung [10 phút]

**Tạo file mới:** `training/models/attention_pooling.py`

```python
"""Shared Attention Pooling module used by multiple model architectures."""
from __future__ import annotations

import torch
from torch import nn, Tensor


class AttentionPooling(nn.Module):
    """Temporal attention pooling that supports masking for padding tokens.

    Args:
        dim: Dimension of input features.

    Input:
        x: (batch, seq_len, dim) — sequence of feature vectors.
        mask: (batch, seq_len) — True for real tokens, False for padding.

    Output:
        pooled: (batch, dim) — weighted sum of input vectors.
        attn_weights: (batch, seq_len) — attention weights.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1, bias=False),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        attn_logits = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -1e4)
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, attn_weights
```

**Cập nhật `improved_lstm.py`:** Import `AttentionPooling` từ module mới thay vì định nghĩa lại.

```python
# training/models/improved_lstm.py — thay dòng 8-28
from training.models.attention_pooling import AttentionPooling
```

---

### Bước 4: Cải tiến `MulTRegressor` [30 phút] ⭐ QUAN TRỌNG NHẤT

**File:** `training/models/mult.py`

Phiên bản cải tiến với: Attention Pooling, Padding Mask, Fusion Head mạnh hơn, và LayerNorm trước fusion.

```python
class MulTRegressor(nn.Module):
    """Multimodal Transformer for sentiment regression.

    Architecture Overview (Improved):
        1. Project each modality to shared d_model dimension
        2. Add positional encoding
        3. Cross-modal attention: 6 directional flows with PADDING MASK
           (T←A, T←V, A←T, A←V, V←T, V←A)
        4. Merge cross-modal outputs per modality (residual sum)
        5. Self-attention transformer encoder per modality
        6. Attention Pooling with padding mask (thay vì lấy timestep cuối)
        7. LayerNorm per modality
        8. Concatenate 3 modalities → Enhanced FC → sentiment score
    """

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        h = config.num_heads
        n_cross = config.num_cross_layers
        n_self = config.num_self_layers
        ffn = config.ffn_dim
        drop = config.attn_dropout

        # --- 1. Projection layers ---
        self.proj_text = nn.Linear(config.text_input_dim, d)
        self.proj_audio = nn.Linear(config.audio_input_dim, d)
        self.proj_vision = nn.Linear(config.vision_input_dim, d)

        # --- 2. Positional encoding ---
        self.pe = PositionalEncoding(d, dropout=drop)

        # --- 3. Cross-Modal Attention blocks (6 flows) ---
        self.cross_t_a = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)
        self.cross_t_v = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)
        self.cross_a_t = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)
        self.cross_a_v = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)
        self.cross_v_t = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)
        self.cross_v_a = CrossModalTransformerBlock(d, h, n_cross, ffn, drop)

        # --- 4. Self-Attention Transformer Encoders ---
        self.self_attn_text = SelfAttentionEncoder(d, h, n_self, ffn, drop)
        self.self_attn_audio = SelfAttentionEncoder(d, h, n_self, ffn, drop)
        self.self_attn_vision = SelfAttentionEncoder(d, h, n_self, ffn, drop)

        # --- 5. Attention Pooling (thay cho last timestep) ---
        self.text_pool = AttentionPooling(d)
        self.audio_pool = AttentionPooling(d)
        self.vision_pool = AttentionPooling(d)

        # --- 6. LayerNorm trước fusion ---
        self.text_ln = nn.LayerNorm(d)
        self.audio_ln = nn.LayerNorm(d)
        self.vision_ln = nn.LayerNorm(d)

        # --- 7. Enhanced Fusion head ---
        fusion_input_dim = d * 3
        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.LayerNorm(config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout * 0.5),
            nn.Linear(config.fusion_hidden_dim // 2, config.output_dim),
        )

    def forward(self, text: Tensor, audio: Tensor, vision: Tensor) -> Tensor:
        """
        Args:
            text:   (batch, seq_len, 768)
            audio:  (batch, seq_len, 74)
            vision: (batch, seq_len, 35)
        Returns:
            (batch,) — sentiment score
        """
        # Tạo padding mask (True = valid, False = padding)
        t_mask = (text.abs().sum(dim=-1) > 1e-6)      # (B, S)
        a_mask = (audio.abs().sum(dim=-1) > 1e-6)     # (B, S)
        v_mask = (vision.abs().sum(dim=-1) > 1e-6)    # (B, S)

        # 1. Project to d_model
        t = self.pe(self.proj_text(text))       # (B, S, d)
        a = self.pe(self.proj_audio(audio))     # (B, S, d)
        v = self.pe(self.proj_vision(vision))   # (B, S, d)

        # 2. Cross-modal attention (6 flows)
        t_with_a = self.cross_t_a(target=t, source=a)
        t_with_v = self.cross_t_v(target=t, source=v)
        a_with_t = self.cross_a_t(target=a, source=t)
        a_with_v = self.cross_a_v(target=a, source=v)
        v_with_t = self.cross_v_t(target=v, source=t)
        v_with_a = self.cross_v_a(target=v, source=a)

        # 3. Merge (residual sum)
        t_merged = t + t_with_a + t_with_v
        a_merged = a + a_with_t + a_with_v
        v_merged = v + v_with_t + v_with_a

        # 4. Self-attention
        t_encoded = self.self_attn_text(t_merged)
        a_encoded = self.self_attn_audio(a_merged)
        v_encoded = self.self_attn_vision(v_merged)

        # 5. Attention Pooling (thay vì [:, -1, :])
        t_repr, _ = self.text_pool(t_encoded, t_mask)
        a_repr, _ = self.audio_pool(a_encoded, a_mask)
        v_repr, _ = self.vision_pool(v_encoded, v_mask)

        # 6. LayerNorm
        t_repr = self.text_ln(t_repr)
        a_repr = self.audio_ln(a_repr)
        v_repr = self.vision_ln(v_repr)

        # 7. Concatenate and predict
        fused = torch.cat([t_repr, a_repr, v_repr], dim=1)
        return self.regressor(fused).squeeze(-1)
```

> **Lưu ý quan trọng:** Nếu muốn truyền `key_padding_mask` cho Cross-Modal Attention, cần sửa thêm `CrossModalAttentionLayer.forward()` để nhận và truyền `source_padding_mask`. Đây là cải tiến nâng cao, có thể thêm sau khi có baseline MulT chạy được.

---

### Bước 5: Cập nhật Config cho MulT [10 phút]

**File:** `training/config_phase1.py`

Cập nhật `Phase1MulTModelConfig` với các giá trị phù hợp hơn:

```python
@dataclass
class Phase1MulTModelConfig:
    """Configuration for the Multimodal Transformer (MulT) model."""
    # Input dimensions (same as MOSEI features)
    text_input_dim: int = 768
    audio_input_dim: int = 74
    vision_input_dim: int = 35

    # Transformer dimensions
    d_model: int = 64              # Kích thước embedding chung
    num_heads: int = 4             # Số attention heads (d_model % num_heads == 0)
    num_cross_layers: int = 4      # ← Tăng từ 3 → 4 (thêm depth cho cross-modal)
    num_self_layers: int = 2       # Giữ nguyên
    ffn_dim: int = 128             # Feed-forward hidden dim
    attn_dropout: float = 0.1     # Attention dropout

    # Fusion head
    fusion_hidden_dim: int = 128   # Fusion MLP hidden dim
    fusion_dropout: float = 0.3    # Fusion dropout
    output_dim: int = 1            # Regression output
```

---

### Bước 6: Cập nhật Training Config cho MulT [10 phút]

MulT cần learning rate thấp hơn LSTM vì Transformer nhạy cảm hơn:

**Trong notebook Colab (hoặc sửa defaults):**
```python
# Siêu tham số đề xuất cho MulT
config.training.learning_rate = 5e-4     # Thấp hơn LSTM (1e-3)
config.training.weight_decay = 1e-3      # Tăng regularization (từ 1e-4)
config.training.batch_size = 32          # Giữ nguyên
config.training.num_epochs = 50          # Giữ nguyên
config.training.patience = 10            # Tăng patience (từ 8) — Transformer hội tụ chậm hơn
config.training.scheduler_patience = 4   # Tăng (từ 3)
config.training.max_grad_norm = 0.5      # Giảm (từ 1.0) — gradient Transformer dễ bùng nổ
```

---

### Bước 7: Tạo Notebook Colab `03_mult_training.ipynb` [30 phút]

**File mới:** `notebooks/03_mult_training.ipynb`

Cấu trúc notebook nên theo mẫu `02_baseline_early_fusion.ipynb` nhưng thay đổi:

```python
# ============ Cell 1: Setup ============
# (Giống notebook baseline: mount Drive, clone repo, install deps)

# ============ Cell 2: Configuration ============
USE_DRIVE_OUTPUTS = True
RESUME_TRAINING = False
RESUME_CHECKPOINT_TYPE = 'last'

# ============ Cell 3: Import & Config ============
from training.config_phase1 import Phase1Config
config = Phase1Config()
config.model_type = "mult"  # ← THAY ĐỔI QUAN TRỌNG

# Apply Colab profile
config.apply_profile("colab")
config.runtime.use_gcs = True

# MulT-specific hyperparameters
config.mult_model.d_model = 64
config.mult_model.num_heads = 4
config.mult_model.num_cross_layers = 4
config.mult_model.num_self_layers = 2
config.mult_model.ffn_dim = 128
config.mult_model.attn_dropout = 0.1
config.mult_model.fusion_hidden_dim = 128
config.mult_model.fusion_dropout = 0.3

# Training hyperparameters
config.training.learning_rate = 5e-4
config.training.weight_decay = 1e-3
config.training.num_epochs = 50
config.training.patience = 10
config.training.scheduler_patience = 4
config.training.max_grad_norm = 0.5
config.training.batch_size = 32

# W&B
config.wandb.enable = True
config.wandb.project = "bcda-phase1"

# ============ Cell 4: Create Model ============
from training.models.mult import MulTRegressor
model = MulTRegressor(config.mult_model)

# In tổng số tham số
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============ Cell 5: Load Data ============
from training.dataset_mosei import create_dataloaders
dataloaders = create_dataloaders(config=config, pkl_path=config.paths.mosei_pkl)

# ============ Cell 6: Train ============
from training.trainer import Phase1Trainer
trainer = Phase1Trainer(model=model, config=config)
summary = trainer.fit(dataloaders["train"], dataloaders["valid"])

# ============ Cell 7: Test ============
test_metrics = trainer.evaluate_and_save(
    dataloaders["test"], split="test", epoch=summary["best_epoch"]
)
print(f"Test Results: {test_metrics}")

# ============ Cell 8: Upload to GCS ============
# (Giống notebook baseline)
```

---

### Bước 8: Cập nhật `models/__init__.py` hoàn chỉnh [2 phút]

```python
from .unimodal_encoder import BiLSTMEncoder
from .early_fusion import EarlyFusionLSTMRegressor
from .improved_lstm import ImprovedLSTMRegressor
from .mult import MulTRegressor
from .attention_pooling import AttentionPooling
```

---

## 6. Cấu Hình Siêu Tham Số Đề Xuất

### 6.1. Bảng so sánh các cấu hình thí nghiệm

| Tham số | Config A (Nhỏ) | Config B (Vừa) ⭐ | Config C (Lớn) |
|:---|:---:|:---:|:---:|
| `d_model` | 40 | **64** | 128 |
| `num_heads` | 4 | **4** | 8 |
| `num_cross_layers` | 3 | **4** | 5 |
| `num_self_layers` | 1 | **2** | 3 |
| `ffn_dim` | 64 | **128** | 256 |
| `fusion_hidden_dim` | 64 | **128** | 256 |
| `attn_dropout` | 0.1 | **0.1** | 0.15 |
| `fusion_dropout` | 0.2 | **0.3** | 0.4 |
| `learning_rate` | 1e-3 | **5e-4** | 1e-4 |
| `weight_decay` | 1e-4 | **1e-3** | 5e-3 |
| `max_grad_norm` | 1.0 | **0.5** | 0.5 |
| `patience` | 8 | **10** | 12 |
| Ước tính params | ~1.5M | **~3M** | ~8M |

> **Khuyến nghị:** Bắt đầu với **Config B (Vừa)** vì cân bằng giữa capacity và tốc độ hội tụ. Nếu vẫn overfitting, chuyển sang Config A. Nếu underfitting, thử Config C.

### 6.2. Chiến lược chống Overfitting cho MulT

1. **Weight Decay cao hơn LSTM:** MulT có nhiều tham số hơn → cần regularization mạnh hơn (1e-3 thay vì 1e-4).
2. **Gradient Clipping thấp:** Transformer gradient dễ bùng nổ → `max_grad_norm=0.5`.
3. **Learning Rate Warmup (tùy chọn nâng cao):** Thêm warmup scheduler cho 2-3 epochs đầu.
4. **Label Smoothing (tùy chọn nâng cao):** Thêm noise nhỏ vào labels để giảm overconfidence.
5. **Dropout đủ mạnh:** `attn_dropout=0.1` + `fusion_dropout=0.3`.

---

## 7. Tạo Notebook Huấn Luyện Colab

### 7.1. Cấu trúc file

```
notebooks/
├── 01_data_exploration.ipynb           # Đã có
├── 02_baseline_early_fusion.ipynb      # Đã có (Baseline LSTM)
├── 02_improved_early_fusion.ipynb      # Đã có (Improved LSTM)
└── 03_mult_training.ipynb              # ← TẠO MỚI
```

### 7.2. Checklist cho notebook

- [ ] Cell 0: README / Mô tả thí nghiệm
- [ ] Cell 1: Mount Google Drive + Clone repo
- [ ] Cell 2: Install dependencies (`pip install wandb`)
- [ ] Cell 3: Configuration (model_type="mult", hyperparameters)
- [ ] Cell 4: Khởi tạo mô hình + In tổng params
- [ ] Cell 5: Load dữ liệu + Kiểm tra shapes
- [ ] Cell 6: Huấn luyện (trainer.fit)
- [ ] Cell 7: Đánh giá trên Test set
- [ ] Cell 8: Upload kết quả lên GCS
- [ ] Cell 9: So sánh với LSTM baselines

---

## 8. Kế Hoạch Xác Minh & Đánh Giá

### 8.1. Smoke Test trước khi train full

Chạy script kiểm tra nhanh **trước khi commit**:

```python
"""Smoke test: Kiểm tra MulT model chạy được forward pass."""
import torch
from training.config_phase1 import Phase1MulTModelConfig
from training.models.mult import MulTRegressor

config = Phase1MulTModelConfig()
model = MulTRegressor(config)

# Tạo dummy input
B, S = 4, 50
text = torch.randn(B, S, 768)
audio = torch.randn(B, S, 74)
vision = torch.randn(B, S, 35)

# Forward pass
output = model(text=text, audio=audio, vision=vision)

assert output.shape == (B,), f"Expected shape (B,), got {output.shape}"
print(f"✅ Forward pass OK. Output shape: {output.shape}")
print(f"✅ Total params: {sum(p.numel() for p in model.parameters()):,}")

# Test backward pass
loss = output.sum()
loss.backward()
print("✅ Backward pass OK.")
```

### 8.2. Mục tiêu đánh giá

So sánh với benchmark MMSA và mô hình LSTM đã train:

| Chỉ số | Baseline LSTM | Improved LSTM | **MulT (Mục tiêu)** | **MMSA Benchmark** |
|:---|:---:|:---:|:---:|:---:|
| Test MAE ↓ | 0.6071 | 0.5859 | **≤ 0.5700** | **0.5593** |
| Test Corr ↑ | 0.6995 | 0.7229 | **≥ 0.7300** | **0.7331** |
| Test Acc-2 ↑ | 81.03% | 81.37% | **≥ 82.0%** | **81.15%** |
| Test Acc-5 ↑ | 49.73% | 51.53% | **≥ 53.0%** | **54.18%** |
| Test Acc-7 ↑ | 48.36% | 49.71% | **≥ 52.0%** | **52.84%** |

### 8.3. Checklist xác minh sau khi train

- [ ] Forward pass smoke test thành công
- [ ] Huấn luyện không crash trong 3 epochs đầu
- [ ] Valid loss có xu hướng giảm trong 5 epochs đầu
- [ ] Test MAE ≤ 0.5700 (tốt hơn Improved LSTM ít nhất 2.7%)
- [ ] Test Corr ≥ 0.7300 (tốt hơn Improved LSTM ít nhất 1.0%)
- [ ] Overfitting ratio (Train/Valid Loss) ≤ 5x ở best epoch
- [ ] Kết quả được log đầy đủ trên W&B
- [ ] Checkpoint được lưu trên GCS
- [ ] So sánh kết quả với MMSA benchmark

### 8.4. Nếu kết quả KHÔNG đạt mục tiêu

| Triệu chứng | Nguyên nhân có thể | Giải pháp |
|:---|:---|:---|
| Valid loss phẳng từ epoch 1 | LR quá nhỏ hoặc d_model quá nhỏ | Tăng LR lên 1e-3 hoặc d_model lên 128 |
| Train loss giảm, Valid loss tăng sớm | Overfitting | Tăng dropout, weight_decay, hoặc giảm d_model |
| Loss NaN/Inf | Gradient explosion | Giảm LR, giảm max_grad_norm xuống 0.1 |
| Valid MAE > 0.60 | Capacity thiếu | Tăng num_cross_layers, d_model |
| Kết quả tệ hơn LSTM | Lỗi code hoặc config sai | Kiểm tra lại Bug 1-4, chạy smoke test |

---

## 9. Tham Khảo & Benchmark

### 9.1. Paper & Code tham khảo

| Tài liệu | Link |
|:---|:---|
| Paper gốc MulT (ACL 2019) | [arXiv:1906.00295](https://arxiv.org/abs/1906.00295) |
| Code tham khảo (MMSA) | [github.com/thuiar/MMSA](https://github.com/thuiar/MMSA) |
| Benchmark MOSEI Results | `data/MSA-Dataset/Git/MMSA/results/result-stat.md` |
| Báo cáo Phase 1 LSTM | `docs/reports/PHASE1_TRAINING_REPORT.md` |
| Lộ trình tổng thể | `docs/research/TRAINING_ROADMAP.md` |

### 9.2. Benchmark MMSA trên MOSEI (Regression)

Trích từ `result-stat.md` — các mô hình liên quan:

| Model | Acc-2 (Has0) | F1 (Has0) | Acc-5 | Acc-7 | MAE | Corr | Data |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ef_lstm | 77.84 | 78.34 | 51.16 | 50.01 | 60.05 | 68.25 | Aligned |
| **mult** | **81.15** | **81.56** | **54.18** | **52.84** | **55.93** | **73.31** | Unaligned |
| misa | 80.67 | 81.12 | 53.63 | 52.05 | 55.75 | 75.15 | Unaligned |
| self_mm | 83.76 | 83.82 | 55.53 | 53.87 | 53.09 | 76.49 | Unaligned |

> **Lưu ý:** Benchmark MulT dùng **Unaligned** data, nhưng dự án của chúng ta dùng **Aligned** data.
> Kết quả có thể khác nhẹ. Mục tiêu ban đầu là đạt ~90% benchmark.

### 9.3. Thứ tự ưu tiên triển khai

```
Ưu tiên 1 (PHẢI LÀM):
   ├── Bước 1: Sửa __init__.py (export MulTRegressor)
   ├── Bước 2: Sửa dataset_mosei.py (validate shapes)
   └── Bước 7: Tạo notebook Colab

Ưu tiên 2 (NÊN LÀM):
   ├── Bước 3: Tách AttentionPooling module
   ├── Bước 4: Cải tiến MulTRegressor (Attention Pooling + LayerNorm)
   └── Bước 5: Cập nhật Config

Ưu tiên 3 (TÙY CHỌN NÂNG CAO):
   ├── Thêm padding mask cho Cross-Modal Attention
   ├── Learning Rate Warmup scheduler
   └── Label Smoothing
```

---

## Tổng Kết

Tài liệu này cung cấp đủ thông tin để một agent code có thể:

1. **Hiểu** kiến trúc MulT và lý do chuyển từ LSTM sang Transformer
2. **Phát hiện** và **sửa** các bug hiện có trong codebase
3. **Triển khai** các cải tiến theo thứ tự ưu tiên rõ ràng
4. **Huấn luyện** mô hình trên Google Colab với config phù hợp
5. **Đánh giá** kết quả và xử lý các tình huống bất ngờ

> **Bước tiếp theo:** Sau khi MulT đạt kết quả tốt trên CMU-MOSEI (Phase 1), chuyển sang Phase 2: Fine-tuning trên dữ liệu tiếng Việt theo hướng dẫn trong `TRAINING_ROADMAP.md`.
