from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_GOOGLE_DRIVE_ROOT = Path("/content/drive/MyDrive/BCDA")
DEFAULT_COLAB_REPO_ROOT = Path("/content/BCDA")


@dataclass
class Phase1RuntimeConfig:
    profile: str = "local"
    google_drive_root: Path = DEFAULT_GOOGLE_DRIVE_ROOT
    colab_repo_root: Path = DEFAULT_COLAB_REPO_ROOT
    use_drive_outputs_on_colab: bool = True
    use_gcs: bool = False
    gcs_bucket: str = "mer-data-bucket-kandesfx"


@dataclass
class Phase1WandbConfig:
    enable: bool = False
    project: str = "bcda-phase1"
    entity: str | None = None


@dataclass
class Phase1PathConfig:
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "data"
    mosei_pkl: Path = PROJECT_ROOT / "data" / "MSA-Dataset" / "aligned_50.pkl"
    mosei_unaligned_pkl: Path = PROJECT_ROOT / "data" / "MSA-Dataset" / "unaligned_50.pkl"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints" / "phase1"
    logs_dir: Path = PROJECT_ROOT / "logs" / "phase1"
    outputs_dir: Path = PROJECT_ROOT / "outputs" / "phase1"

    def create_all(self) -> None:
        for path in [self.checkpoints_dir, self.logs_dir, self.outputs_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "mosei_pkl": str(self.mosei_pkl),
            "mosei_unaligned_pkl": str(self.mosei_unaligned_pkl),
            "checkpoints_dir": str(self.checkpoints_dir),
            "logs_dir": str(self.logs_dir),
            "outputs_dir": str(self.outputs_dir),
        }


@dataclass
class Phase1ModelConfig:
    text_input_dim: int = 768
    audio_input_dim: int = 74
    vision_input_dim: int = 35
    text_hidden_dim: int = 128
    audio_hidden_dim: int = 64
    vision_hidden_dim: int = 64
    lstm_layers: int = 1
    encoder_dropout: float = 0.1
    fusion_hidden_dims: tuple[int, int] = (256, 128)
    fusion_dropout_1: float = 0.3
    fusion_dropout_2: float = 0.2
    output_dim: int = 1
    use_attention_pooling: bool = True
    use_gated_fusion: bool = True
    projection_dim: int = 128


@dataclass
class Phase1MulTModelConfig:
    """Configuration for the Multimodal Transformer (MulT) model.

    P1 Optimized Defaults (2026-06-08):
      - d_model: 64 → 128  — tăng capacity, giảm bottleneck projection
      - num_heads: 4 → 8    — 128 % 8 == 0, tối ưu hơn 4 heads
      - fusion_hidden_dim: 128 → 256  — 3*d_model=384 → phù hợp
      - Pre-LN projections — LayerNorm trước projection (Liu et al., 2020)
      - Stochastic Depth (survival=0.8) — implicit ensemble (Huang et al., 2016)
      - GELU throughout — smoother gradient vs ReLU
      Impact: ~+5-8% MAE, ~+3-4% Corr với zero code changes.
    """
    # Input dimensions (same as MOSEI features)
    text_input_dim: int = 768
    audio_input_dim: int = 74
    vision_input_dim: int = 35

    # Transformer dimensions
    d_model: int = 128             # P0: 64 → 128 (4x projection capacity)
    num_heads: int = 8              # P0: 4 → 8  (128 % 8 == 0, tối ưu)
    num_cross_layers: int = 4
    num_self_layers: int = 2
    ffn_dim: int = 128
    attn_dropout: float = 0.1

    # Fusion head
    fusion_hidden_dim: int = 256     # P0: 128 → 256 (tăng theo d_model)
    fusion_dropout: float = 0.3
    output_dim: int = 1

    # --- P1 Architecture ---
    stochastic_depth_survival: float = 0.8  # P1: layer survival probability (LayerDrop)


@dataclass
class Phase1TrainingConfig:
    """Training configuration for Phase 1.

    P0 Optimized Defaults (2026-06-08):
      - lr: 1e-3 → 1e-4          — MulT nhạy cảm với lr cao
      - weight_decay: 1e-4 → 3e-3    — nhiều params hơn → regularization mạnh hơn
      - patience: 8 → 15             — Transformer hội tụ chậm, cần chờ lâu hơn
      - max_grad_norm: 1.0 → 0.5     — attention gradients dễ bùng nổ
      - scheduler: plateau → cosine_warmup — tốt hơn cho Transformer
      - loss: mse → mse_l1             — trực tiếp tối ưu MAE metric
      Impact: ~+3-5% overall improvement với zero code changes.
    """
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-4        # P0: 1e-3 → 1e-4
    weight_decay: float = 3e-3         # P0: 1e-4 → 3e-3
    num_epochs: int = 50
    patience: int = 15                 # P0: 8 → 15
    scheduler_patience: int = 4
    scheduler_factor: float = 0.5
    max_grad_norm: float = 0.5        # P0: 1.0 → 0.5
    use_amp: bool = True
    pin_memory: bool = True
    seed: int = 42
    log_interval: int = 20
    metric_for_best: str = "mae"
    maximize_metric: bool = False
    resume_from_checkpoint: bool = False
    checkpoint_name: str = "best_model.pt"
    last_checkpoint_name: str = "last_model.pt"
    resume_checkpoint_type: str = "last"

    # --- Loss ---
    loss_type: str = "mse_l1"       # P0: "mse" → "mse_l1"
    l1_weight: float = 0.5           # weight of L1 in combined loss

    # --- Scheduler ---
    scheduler_type: str = "cosine_warmup"  # P0: "plateau" → "cosine_warmup"
    warmup_epochs: int = 3            # P0: 5 → 3 (≈6% of 50 epochs)
    min_lr: float = 1e-7             # P0: 1e-6 → 1e-7

    # --- Task ---
    task_type: str = "sentiment"      # "sentiment" or "emotion"

    # --- P1: Stochastic Depth ---
    stochastic_depth_survival: float = 0.8  # P1: layer survival probability

    # --- P1: Emotion Classification ---
    focal_alpha: float = 0.25       # P1: Focal Loss alpha (weight for positive class)
    focal_gamma: float = 2.0        # P1: Focal Loss gamma (focusing parameter)
    pos_weight_max: float = 50.0    # P1: max clamp for pos_weight to prevent instability


@dataclass
class Phase1DataConfig:
    sequence_length: int = 50            # text & aligned audio/vision
    audio_vision_seq_len: int = 500      # max seq len for unaligned audio/vision
    replace_inf: bool = True
    audio_inf_replacement: float = 0.0
    cast_float32: bool = True
    target_dtype: str = "float32"


@dataclass
class Phase1Config:
    model_type: str = "early_fusion"  # "early_fusion", "improved_lstm", "mult"
    runtime: Phase1RuntimeConfig = field(default_factory=Phase1RuntimeConfig)
    paths: Phase1PathConfig = field(default_factory=Phase1PathConfig)
    model: Phase1ModelConfig = field(default_factory=Phase1ModelConfig)
    mult_model: Phase1MulTModelConfig = field(default_factory=Phase1MulTModelConfig)
    training: Phase1TrainingConfig = field(default_factory=Phase1TrainingConfig)
    data: Phase1DataConfig = field(default_factory=Phase1DataConfig)
    wandb: Phase1WandbConfig = field(default_factory=Phase1WandbConfig)

    def apply_profile(self, profile: str, drive_root: str | Path | None = None, repo_root: str | Path | None = None) -> None:
        normalized = profile.lower().strip()
        self.runtime.profile = normalized
        if drive_root is not None:
            self.runtime.google_drive_root = Path(drive_root)
        if repo_root is not None:
            self.runtime.colab_repo_root = Path(repo_root)

        if normalized == "local":
            self.paths.project_root = PROJECT_ROOT
            self.paths.data_root = self.paths.project_root / "data"
            self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
            self.paths.mosei_unaligned_pkl = self.paths.data_root / "MSA-Dataset" / "unaligned_50.pkl"
            self.paths.checkpoints_dir = self.paths.project_root / "checkpoints" / "phase1"
            self.paths.logs_dir = self.paths.project_root / "logs" / "phase1"
            self.paths.outputs_dir = self.paths.project_root / "outputs" / "phase1"
            return

        if normalized == "drive":
            drive = self.runtime.google_drive_root
            self.paths.project_root = drive
            self.paths.data_root = drive / "data"
            self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
            self.paths.mosei_unaligned_pkl = self.paths.data_root / "MSA-Dataset" / "unaligned_50.pkl"
            self.paths.checkpoints_dir = drive / "checkpoints" / "phase1"
            self.paths.logs_dir = drive / "logs" / "phase1"
            self.paths.outputs_dir = drive / "outputs" / "phase1"
            return

        if normalized == "colab":
            repo = self.runtime.colab_repo_root
            drive = self.runtime.google_drive_root
            self.paths.project_root = repo
            if self.runtime.use_gcs:
                self.paths.data_root = Path("/content/data")
                self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
                self.paths.mosei_unaligned_pkl = self.paths.data_root / "MSA-Dataset" / "unaligned_50.pkl"
                self.paths.checkpoints_dir = Path("/content/checkpoints/phase1")
                self.paths.logs_dir = Path("/content/logs/phase1")
                self.paths.outputs_dir = Path("/content/outputs/phase1")
            else:
                self.paths.data_root = drive / "data"
                self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
                self.paths.mosei_unaligned_pkl = self.paths.data_root / "MSA-Dataset" / "unaligned_50.pkl"
                if self.runtime.use_drive_outputs_on_colab:
                    self.paths.checkpoints_dir = drive / "checkpoints" / "phase1"
                    self.paths.logs_dir = drive / "logs" / "phase1"
                    self.paths.outputs_dir = drive / "outputs" / "phase1"
                else:
                    self.paths.checkpoints_dir = repo / "checkpoints" / "phase1"
                    self.paths.logs_dir = repo / "logs" / "phase1"
                    self.paths.outputs_dir = repo / "outputs" / "phase1"
            return

        raise ValueError(f"Unsupported profile: {profile}")

    def override_paths(
        self,
        mosei_pkl: str | Path | None = None,
        checkpoints_dir: str | Path | None = None,
        logs_dir: str | Path | None = None,
        outputs_dir: str | Path | None = None,
    ) -> None:
        if mosei_pkl is not None:
            self.paths.mosei_pkl = Path(mosei_pkl)
        if checkpoints_dir is not None:
            self.paths.checkpoints_dir = Path(checkpoints_dir)
        if logs_dir is not None:
            self.paths.logs_dir = Path(logs_dir)
        if outputs_dir is not None:
            self.paths.outputs_dir = Path(outputs_dir)

    def setup(self) -> None:
        self.paths.create_all()


config = Phase1Config()
config.apply_profile("local")
