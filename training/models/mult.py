"""
MulT — Multimodal Transformer for Sentiment Regression.

Implements the Cross-Modal Attention mechanism from:
    "Multimodal Transformer for Unaligned Multimodal Language Sequences"
    (Tsai et al., ACL 2019)

Supports both:
  - Aligned mode  (aligned_50.pkl):   all 3 modalities share seq_len=50
  - Unaligned mode (unaligned_50.pkl): audio/vision have seq_len=500,
    with audio_lengths / vision_lengths indicating true lengths.
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

from training.models.attention_pooling import AttentionPooling


# ---------------------------------------------------------------------------
# Helper: convert length integers → boolean padding mask
# ---------------------------------------------------------------------------

def lengths_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """Convert a 1-D lengths tensor to a boolean valid-positions mask.

    Args:
        lengths: (batch,) int64 — actual sequence length per sample
        max_len: total padded sequence length
    Returns:
        mask: (batch, max_len) bool — True where position is valid (not padding)
    """
    # arange: (1, max_len)  vs lengths: (batch, 1)
    return torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Injects positional information via sine / cosine waves.

    Transformer has no built-in notion of token order (unlike LSTM),
    so we add a unique positional signal to each timestep.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)     # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)         # even indices
        pe[:, 1::2] = torch.cos(position * div_term)         # odd indices
        pe = pe.unsqueeze(0)                                  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Cross-Modal Attention Layer
# ---------------------------------------------------------------------------

class CrossModalAttentionLayer(nn.Module):
    """One layer of cross-modal attention + feed-forward.

    Query comes from the **target** modality.
    Key and Value come from the **source** modality.
    This lets the target modality "look at" the source modality.

    Architecture per layer:
        x_target = LayerNorm(x_target + MultiHeadAttn(Q=x_target, KV=x_source))
        x_target = LayerNorm(x_target + FFN(x_target))
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, target: Tensor, source: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        """
        target: (batch, seq_len, d_model) — modality that asks questions (Query)
        source: (batch, seq_len, d_model) — modality that provides answers (Key, Value)
        key_padding_mask: (batch, seq_len) — True for positions to be ignored in attention
        """
        # Cross-attention: target queries source
        attn_out, _ = self.cross_attn(
            query=target,
            key=source,
            value=source,
            key_padding_mask=key_padding_mask,
        )
        target = self.norm1(target + attn_out)      # residual + layer norm

        # Feed-forward
        ffn_out = self.ffn(target)
        target = self.norm2(target + ffn_out)        # residual + layer norm

        return target


# ---------------------------------------------------------------------------
# Cross-Modal Transformer Block (stacks N cross-attention layers)
# ---------------------------------------------------------------------------

class CrossModalTransformerBlock(nn.Module):
    """Stack of N CrossModalAttentionLayers for one (target ← source) pair."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, target: Tensor, source: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        for layer in self.layers:
            target = layer(target, source, key_padding_mask=key_padding_mask)
        return target


# ---------------------------------------------------------------------------
# Self-Attention Transformer Encoder
# ---------------------------------------------------------------------------

class SelfAttentionEncoder(nn.Module):
    """Standard Transformer Encoder for temporal self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# MulT Regressor (main model)
# ---------------------------------------------------------------------------

class MulTRegressor(nn.Module):
    """Multimodal Transformer for sentiment regression.

    Architecture Overview (Improved):
        1. Project each modality to shared d_model dimension
        2. Add positional encoding
        3. Cross-modal attention: 6 directional flows
           (T←A, T←V, A←T, A←V, V←T, V←A)
        4. Merge cross-modal outputs per modality (residual sum)
        5. Self-attention transformer encoder per modality
        6. Attention Pooling with padding mask (instead of last timestep)
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

        # --- 5. Attention Pooling (instead of last timestep) ---
        self.text_pool = AttentionPooling(d)
        self.audio_pool = AttentionPooling(d)
        self.vision_pool = AttentionPooling(d)

        # --- 6. LayerNorm before fusion ---
        self.text_ln = nn.LayerNorm(d)
        self.audio_ln = nn.LayerNorm(d)
        self.vision_ln = nn.LayerNorm(d)

        # --- 7. Enhanced Fusion head ---
        fusion_input_dim = d * 3
        self.output_dim = config.output_dim
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

    @staticmethod
    def _ensure_valid_mask(mask: Tensor) -> Tensor:
        """Guarantee at least one True (valid) position per row.

        Root cause protection: MOSEI vision features are all-zero for samples
        where no face was detected (482 train / 41 valid / 174 test samples).
        An all-False mask causes nn.MultiheadAttention to compute
        softmax([-inf, -inf, ...]) = NaN, poisoning the entire batch.

        Fix: force position 0 to be valid for any all-masked rows.
        This is safe — position 0 holds a real (zero) feature vector whose
        contribution will be down-weighted by attention scores anyway.
        """
        all_invalid = ~mask.any(dim=-1, keepdim=True)   # (B, 1)
        if all_invalid.any():
            mask = mask.clone()
            mask[:, 0] = mask[:, 0] | all_invalid.squeeze(-1)
        return mask

    def forward(
        self,
        text: Tensor,
        audio: Tensor,
        vision: Tensor,
        audio_lengths: Tensor | None = None,
        vision_lengths: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            text:           (batch, T,   768)  — text features
            audio:          (batch, A,    74)  — audio features  (A=50 aligned / A=500 unaligned)
            vision:         (batch, V,    35)  — vision features (V=50 aligned / V=500 unaligned)
            audio_lengths:  (batch,) int64 optional — actual audio frames (unaligned only)
            vision_lengths: (batch,) int64 optional — actual vision frames (unaligned only)
        Returns:
            (batch,) — sentiment score
        """
        # --- Padding masks (True = valid token, False = padding) ---
        # Aligned: detect zeros. Unaligned: use provided lengths for precision.
        t_mask = self._ensure_valid_mask(text.abs().sum(dim=-1) > 1e-6)   # (B, T)

        if audio_lengths is not None:
            a_mask = self._ensure_valid_mask(
                lengths_to_mask(audio_lengths, audio.size(1))              # (B, A)
            )
        else:
            a_mask = self._ensure_valid_mask(audio.abs().sum(dim=-1) > 1e-6)  # (B, A)

        if vision_lengths is not None:
            v_mask = self._ensure_valid_mask(
                lengths_to_mask(vision_lengths, vision.size(1))            # (B, V)
            )
        else:
            v_mask = self._ensure_valid_mask(vision.abs().sum(dim=-1) > 1e-6) # (B, V)

        # 1. Project to d_model
        t = self.pe(self.proj_text(text))       # (B, T, d)
        a = self.pe(self.proj_audio(audio))     # (B, A, d)
        v = self.pe(self.proj_vision(vision))   # (B, V, d)

        # 2. Cross-modal attention (6 flows with source key_padding_mask)
        #    Q and K/V can have different sequence lengths — nn.MultiheadAttention
        #    handles this natively (output shape = Q shape).
        t_with_a = self.cross_t_a(target=t, source=a, key_padding_mask=~a_mask)  # (B, T, d)
        t_with_v = self.cross_t_v(target=t, source=v, key_padding_mask=~v_mask)  # (B, T, d)
        a_with_t = self.cross_a_t(target=a, source=t, key_padding_mask=~t_mask)  # (B, A, d)
        a_with_v = self.cross_a_v(target=a, source=v, key_padding_mask=~v_mask)  # (B, A, d)
        v_with_t = self.cross_v_t(target=v, source=t, key_padding_mask=~t_mask)  # (B, V, d)
        v_with_a = self.cross_v_a(target=v, source=a, key_padding_mask=~a_mask)  # (B, V, d)

        # 3. Merge (residual sum)
        t_merged = t + t_with_a + t_with_v     # (B, T, d)
        a_merged = a + a_with_t + a_with_v     # (B, A, d)
        v_merged = v + v_with_t + v_with_a     # (B, V, d)

        # 4. Self-attention (with padding masks)
        t_encoded = self.self_attn_text(t_merged,  key_padding_mask=~t_mask)  # (B, T, d)
        a_encoded = self.self_attn_audio(a_merged, key_padding_mask=~a_mask)  # (B, A, d)
        v_encoded = self.self_attn_vision(v_merged, key_padding_mask=~v_mask) # (B, V, d)

        # 5. Attention Pooling → fixed-size representation per modality
        t_repr, _ = self.text_pool(t_encoded,  t_mask)   # (B, d)
        a_repr, _ = self.audio_pool(a_encoded, a_mask)   # (B, d)
        v_repr, _ = self.vision_pool(v_encoded, v_mask)  # (B, d)

        # 6. LayerNorm
        t_repr = self.text_ln(t_repr)
        a_repr = self.audio_ln(a_repr)
        v_repr = self.vision_ln(v_repr)

        # 7. Concatenate and predict
        fused = torch.cat([t_repr, a_repr, v_repr], dim=1)  # (B, 3*d)
        out = self.regressor(fused)                           # (B, output_dim)
        # Squeeze for regression (output_dim=1), keep shape for multi-label (output_dim=6)
        return out.squeeze(-1) if self.output_dim == 1 else out
