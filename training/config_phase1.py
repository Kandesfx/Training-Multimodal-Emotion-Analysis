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


@dataclass
class Phase1TrainingConfig:
    batch_size: int = 32
    num_workers: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 8
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    max_grad_norm: float = 1.0
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


@dataclass
class Phase1DataConfig:
    sequence_length: int = 50
    replace_inf: bool = True
    audio_inf_replacement: float = 0.0
    cast_float32: bool = True
    target_dtype: str = "float32"


@dataclass
class Phase1Config:
    runtime: Phase1RuntimeConfig = field(default_factory=Phase1RuntimeConfig)
    paths: Phase1PathConfig = field(default_factory=Phase1PathConfig)
    model: Phase1ModelConfig = field(default_factory=Phase1ModelConfig)
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
            self.paths.checkpoints_dir = self.paths.project_root / "checkpoints" / "phase1"
            self.paths.logs_dir = self.paths.project_root / "logs" / "phase1"
            self.paths.outputs_dir = self.paths.project_root / "outputs" / "phase1"
            return

        if normalized == "drive":
            drive = self.runtime.google_drive_root
            self.paths.project_root = drive
            self.paths.data_root = drive / "data"
            self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
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
                self.paths.checkpoints_dir = Path("/content/checkpoints/phase1")
                self.paths.logs_dir = Path("/content/logs/phase1")
                self.paths.outputs_dir = Path("/content/outputs/phase1")
            else:
                self.paths.data_root = drive / "data"
                self.paths.mosei_pkl = self.paths.data_root / "MSA-Dataset" / "aligned_50.pkl"
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
