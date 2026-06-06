"""
MulT — Multimodal Transformer for Sentiment Regression.

Implements the Cross-Modal Attention mechanism from:
    "Multimodal Transformer for Unaligned Multimodal Language Sequences"
    (Tsai et al., ACL 2019)

Adapted for CMU-MOSEI aligned features (Phase 1 Pre-training).
"""
from __future__ import annotations

import math

import torch
from torch import nn, Tensor

from training.models.attention_pooling import AttentionPooling


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

        # 2. Cross-modal attention (6 flows with source padding masks)
        t_with_a = self.cross_t_a(target=t, source=a, key_padding_mask=~a_mask)
        t_with_v = self.cross_t_v(target=t, source=v, key_padding_mask=~v_mask)
        a_with_t = self.cross_a_t(target=a, source=t, key_padding_mask=~t_mask)
        a_with_v = self.cross_a_v(target=a, source=v, key_padding_mask=~v_mask)
        v_with_t = self.cross_v_t(target=v, source=t, key_padding_mask=~t_mask)
        v_with_a = self.cross_v_a(target=v, source=a, key_padding_mask=~a_mask)

        # 3. Merge (residual sum)
        t_merged = t + t_with_a + t_with_v
        a_merged = a + a_with_t + a_with_v
        v_merged = v + v_with_t + v_with_a

        # 4. Self-attention (with padding masks)
        t_encoded = self.self_attn_text(t_merged, key_padding_mask=~t_mask)
        a_encoded = self.self_attn_audio(a_merged, key_padding_mask=~a_mask)
        v_encoded = self.self_attn_vision(v_merged, key_padding_mask=~v_mask)

        # 5. Attention Pooling (instead of [:, -1, :])
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
