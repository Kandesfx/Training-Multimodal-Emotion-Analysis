from __future__ import annotations

import torch
from torch import nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=effective_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        return torch.cat([forward_hidden, backward_hidden], dim=1)
