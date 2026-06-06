from __future__ import annotations

import torch
from torch import nn
from training.config_phase1 import Phase1ModelConfig
from training.models.attention_pooling import AttentionPooling


class GatedFusion(nn.Module):
    """Dynamic Gated Multimodal Fusion mechanism to scale modality inputs."""
    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int, projection_dim: int, dropout: float = 0.3):
        super().__init__()
        # Projections to common space with LayerNorm and Dropout
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gate network based on joint context
        concat_dim = 3 * projection_dim
        self.gate_text = nn.Sequential(
            nn.Linear(concat_dim, projection_dim),
            nn.Sigmoid()
        )
        self.gate_audio = nn.Sequential(
            nn.Linear(concat_dim, projection_dim),
            nn.Sigmoid()
        )
        self.gate_vision = nn.Sequential(
            nn.Linear(concat_dim, projection_dim),
            nn.Sigmoid()
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        # text: [B, text_dim], audio: [B, audio_dim], vision: [B, vision_dim]
        t_proj = self.text_proj(text)
        a_proj = self.audio_proj(audio)
        v_proj = self.vision_proj(vision)

        concat = torch.cat([t_proj, a_proj, v_proj], dim=-1) # [B, 3 * projection_dim]

        g_t = self.gate_text(concat)
        g_a = self.gate_audio(concat)
        g_v = self.gate_vision(concat)

        t_gated = t_proj * g_t
        a_gated = a_proj * g_a
        v_gated = v_proj * g_v

        # Concatenate gated modalities
        return torch.cat([t_gated, a_gated, v_gated], dim=-1) # [B, 3 * projection_dim]


class ImprovedLSTMRegressor(nn.Module):
    """Improved Early Fusion LSTM model with LayerNorm, Attention Pooling, and Gated Fusion."""
    def __init__(self, model_config: Phase1ModelConfig):
        super().__init__()
        self.config = model_config

        # 1. Recurrent Encoders
        self.text_lstm = nn.LSTM(
            input_size=model_config.text_input_dim,
            hidden_size=model_config.text_hidden_dim,
            num_layers=model_config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=model_config.encoder_dropout if model_config.lstm_layers > 1 else 0.0,
        )
        self.audio_lstm = nn.LSTM(
            input_size=model_config.audio_input_dim,
            hidden_size=model_config.audio_hidden_dim,
            num_layers=model_config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=model_config.encoder_dropout if model_config.lstm_layers > 1 else 0.0,
        )
        self.vision_lstm = nn.LSTM(
            input_size=model_config.vision_input_dim,
            hidden_size=model_config.vision_hidden_dim,
            num_layers=model_config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=model_config.encoder_dropout if model_config.lstm_layers > 1 else 0.0,
        )

        # 2. Attention Pooling (optional, default: True)
        # Bidirectional hidden output dim is 2 * hidden_dim
        text_enc_dim = model_config.text_hidden_dim * 2
        audio_enc_dim = model_config.audio_hidden_dim * 2
        vision_enc_dim = model_config.vision_hidden_dim * 2

        if getattr(model_config, "use_attention_pooling", True):
            self.text_attn = AttentionPooling(text_enc_dim)
            self.audio_attn = AttentionPooling(audio_enc_dim)
            self.vision_attn = AttentionPooling(vision_enc_dim)
        else:
            self.text_attn = None
            self.audio_attn = None
            self.vision_attn = None

        # 3. Layer Normalization for stability before fusion
        self.text_ln = nn.LayerNorm(text_enc_dim)
        self.audio_ln = nn.LayerNorm(audio_enc_dim)
        self.vision_ln = nn.LayerNorm(vision_enc_dim)

        # 4. Multimodal Fusion Layer
        if getattr(model_config, "use_gated_fusion", True):
            proj_dim = getattr(model_config, "projection_dim", 128)
            self.fusion = GatedFusion(
                text_dim=text_enc_dim,
                audio_dim=audio_enc_dim,
                vision_dim=vision_enc_dim,
                projection_dim=proj_dim,
                dropout=model_config.encoder_dropout
            )
            fusion_input_dim = 3 * proj_dim
        else:
            self.fusion = None
            fusion_input_dim = text_enc_dim + audio_enc_dim + vision_enc_dim

        # 5. Regressor MLP
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
        # LSTM outputs: [B, T, 2 * hidden_dim]
        t_out, (t_hid, _) = self.text_lstm(text)
        a_out, (a_hid, _) = self.audio_lstm(audio)
        v_out, (v_hid, _) = self.vision_lstm(vision)

        # Attention Pooling or standard last hidden state
        if self.text_attn is not None:
            # Mask out padding tokens (padded locations are typically all zeros)
            t_mask = (text.abs().sum(dim=-1) > 1e-6)
            a_mask = (audio.abs().sum(dim=-1) > 1e-6)
            v_mask = (vision.abs().sum(dim=-1) > 1e-6)

            t_repr, _ = self.text_attn(t_out, t_mask)
            a_repr, _ = self.audio_attn(a_out, a_mask)
            v_repr, _ = self.vision_attn(v_out, v_mask)
        else:
            # Revert to last layer's final bidirectional hidden states
            t_repr = torch.cat([t_hid[-2], t_hid[-1]], dim=-1)
            a_repr = torch.cat([a_hid[-2], a_hid[-1]], dim=-1)
            v_repr = torch.cat([v_hid[-2], v_hid[-1]], dim=-1)

        # Apply LayerNorm
        t_repr = self.text_ln(t_repr)
        a_repr = self.audio_ln(a_repr)
        v_repr = self.vision_ln(v_repr)

        # Fusion
        if self.fusion is not None:
            fused = self.fusion(t_repr, a_repr, v_repr)
        else:
            fused = torch.cat([t_repr, a_repr, v_repr], dim=-1)

        # Regressor output: [B, 1] -> squeeze to [B]
        return self.regressor(fused).squeeze(-1)
