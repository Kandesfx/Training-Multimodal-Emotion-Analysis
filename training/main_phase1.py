from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config_phase1 import Phase1Config, config as default_config
from training.dataset_mosei import create_dataloaders
from training.trainer import Phase1Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Phase 1 models on CMU-MOSEI aligned features"
    )
    parser.add_argument("--pkl-path", type=str, default=None, help="Optional override path to aligned_50.pkl")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--task-type", type=str, default=None,
        choices=["sentiment", "emotion"],
        help="Task type: 'sentiment' (regression) or 'emotion' (multi-label classification)"
    )
    parser.add_argument(
        "--loss-type", type=str, default=None,
        choices=["mse", "mse_l1", "bce", "focal"],
        help="Loss function type"
    )
    parser.add_argument(
        "--model-type", type=str, default=None,
        choices=["early_fusion", "improved_lstm", "mult"],
        help="Model architecture"
    )
    return parser


def apply_overrides(cfg: Phase1Config, args: argparse.Namespace) -> Phase1Config:
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
    if args.num_workers is not None:
        cfg.training.num_workers = args.num_workers
    if args.learning_rate is not None:
        cfg.training.learning_rate = args.learning_rate
    if args.pkl_path is not None:
        cfg.paths.mosei_pkl = Path(args.pkl_path)
    if args.task_type is not None:
        cfg.training.task_type = args.task_type
    if args.loss_type is not None:
        cfg.training.loss_type = args.loss_type
    if args.model_type is not None:
        cfg.model_type = args.model_type
    return cfg


def _sync_mult_stochastic_depth(cfg: Phase1Config) -> None:
    """Sync stochastic_depth_survival from training config to MulT model config.

    This lets the training config control the value while keeping a sensible
    default (0.8) in Phase1MulTModelConfig for standalone use.
    """
    cfg.mult_model.stochastic_depth_survival = cfg.training.stochastic_depth_survival


def _sync_output_dim(cfg: Phase1Config) -> None:
    """Auto-set output_dim based on task_type.

    Emotion task requires output_dim=6 (6 emotions).
    Sentiment task uses output_dim=1 (scalar regression).
    """
    if cfg.model_type == "mult":
        if cfg.training.task_type == "emotion":
            cfg.mult_model.output_dim = 6
        else:
            cfg.mult_model.output_dim = 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = apply_overrides(default_config, args)
    cfg.setup()
    _sync_mult_stochastic_depth(cfg)
    _sync_output_dim(cfg)

    dataloaders = create_dataloaders(config=cfg, pkl_path=cfg.paths.mosei_pkl)
    
    if cfg.model_type == "early_fusion":
        from training.models.early_fusion import EarlyFusionLSTMRegressor
        model = EarlyFusionLSTMRegressor(cfg.model)
    elif cfg.model_type == "improved_lstm":
        from training.models.improved_lstm import ImprovedLSTMRegressor
        model = ImprovedLSTMRegressor(cfg.model)
    elif cfg.model_type == "mult":
        from training.models.mult import MulTRegressor
        model = MulTRegressor(cfg.mult_model)
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    trainer = Phase1Trainer(model=model, config=cfg)

    summary = trainer.fit(dataloaders["train"], dataloaders["valid"])
    test_metrics = trainer.evaluate_and_save(dataloaders["test"], split="test", epoch=summary["best_epoch"])
    output = {"training": summary, "test": test_metrics}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
