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
