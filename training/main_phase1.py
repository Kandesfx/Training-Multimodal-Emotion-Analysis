from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from training.config_phase1 import Phase1Config, config as default_config
from training.dataset_mosei import create_dataloaders
from training.models.early_fusion import EarlyFusionLSTMRegressor
from training.trainer import Phase1Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 1 Early Fusion LSTM on CMU-MOSEI aligned features")
    parser.add_argument("--pkl-path", type=str, default=None, help="Optional override path to aligned_50.pkl")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
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
    return cfg


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = apply_overrides(default_config, args)
    cfg.setup()

    dataloaders = create_dataloaders(config=cfg, pkl_path=cfg.paths.mosei_pkl)
    model = EarlyFusionLSTMRegressor(cfg.model)
    trainer = Phase1Trainer(model=model, config=cfg)

    summary = trainer.fit(dataloaders["train"], dataloaders["valid"])
    test_metrics = trainer.evaluate_and_save(dataloaders["test"], split="test", epoch=summary["best_epoch"])
    output = {"training": summary, "test": test_metrics}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
