from __future__ import annotations

import torch
from torch import nn

from training.config_phase1 import Phase1ModelConfig
from training.models.unimodal_encoder import BiLSTMEncoder


class EarlyFusionLSTMRegressor(nn.Module):
    def __init__(self, model_config: Phase1ModelConfig):
        super().__init__()
        self.text_encoder = BiLSTMEncoder(
            input_dim=model_config.text_input_dim,
            hidden_dim=model_config.text_hidden_dim,
            num_layers=model_config.lstm_layers,
            dropout=model_config.encoder_dropout,
        )
        self.audio_encoder = BiLSTMEncoder(
            input_dim=model_config.audio_input_dim,
            hidden_dim=model_config.audio_hidden_dim,
            num_layers=model_config.lstm_layers,
            dropout=model_config.encoder_dropout,
        )
        self.vision_encoder = BiLSTMEncoder(
            input_dim=model_config.vision_input_dim,
            hidden_dim=model_config.vision_hidden_dim,
            num_layers=model_config.lstm_layers,
            dropout=model_config.encoder_dropout,
        )

        fusion_input_dim = (model_config.text_hidden_dim * 2) + (model_config.audio_hidden_dim * 2) + (model_config.vision_hidden_dim * 2)
        hidden_1, hidden_2 = model_config.fusion_hidden_dims

        self.regressor = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(model_config.fusion_dropout_1),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Dropout(model_config.fusion_dropout_2),
            nn.Linear(hidden_2, model_config.output_dim),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        text_repr = self.text_encoder(text)
        audio_repr = self.audio_encoder(audio)
        vision_repr = self.vision_encoder(vision)
        fused = torch.cat([text_repr, audio_repr, vision_repr], dim=1)
        return self.regressor(fused).squeeze(-1)
