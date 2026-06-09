#!/usr/bin/env python3
"""
Evaluate a trained Phase 1 checkpoint on all splits (train / valid / test).

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint "checkpoints/phase1/best_model.pt" \
        --task-type emotion \
        --model-type mult \
        --pkl-path data/MSA-Dataset/aligned_50.pkl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config_phase1 import Phase1Config, config as default_config
from training.dataset_mosei import create_dataloaders
from training.evaluator import compute_metrics
from training.evaluator_emotion import compute_emotion_metrics


# ---------------------------------------------------------------------------
# Model factory (must match the architecture used during training)
# ---------------------------------------------------------------------------

def build_model(cfg: Phase1Config) -> nn.Module:
    """Instantiate the model matching the checkpoint's architecture."""
    if cfg.model_type == "early_fusion":
        from training.models.early_fusion import EarlyFusionLSTMRegressor
        return EarlyFusionLSTMRegressor(cfg.model)
    elif cfg.model_type == "improved_lstm":
        from training.models.improved_lstm import ImprovedLSTMRegressor
        return ImprovedLSTMRegressor(cfg.model)
    elif cfg.model_type == "mult":
        from training.models.mult import MulTRegressor
        return MulTRegressor(cfg.mult_model)
    else:
        raise ValueError(f"Unsupported model_type: {cfg.model_type!r}")


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_split(
    model: nn.Module,
    data_loader,
    device: torch.device,
    task_type: str,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    """Run inference and compute metrics for one split.

    Returns: (avg_loss, metrics_dict, all_preds, all_labels)
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in data_loader:
            text = batch["text"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            vision = batch["vision"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            audio_lengths = batch.get("audio_len")
            audio_lengths = audio_lengths.to(device, non_blocking=True) if audio_lengths is not None else None
            vision_lengths = batch.get("vision_len")
            vision_lengths = vision_lengths.to(device, non_blocking=True) if vision_lengths is not None else None

            preds = model(
                text=text, audio=audio, vision=vision,
                audio_lengths=audio_lengths, vision_lengths=vision_lengths,
            )

            loss = criterion(preds, labels)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Compute metrics
    if task_type == "emotion":
        metrics = compute_emotion_metrics(y_true, y_pred)
        avg_loss = float(loss.item())
    else:
        metrics = compute_metrics(y_true, y_pred)
        avg_loss = float(loss.item())

    return avg_loss, metrics, y_pred, y_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1 checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--task-type", type=str, default="sentiment",
                        choices=["sentiment", "emotion"])
    parser.add_argument("--model-type", type=str, default="mult",
                        choices=["early_fusion", "improved_lstm", "mult"])
    parser.add_argument("--pkl-path", type=str, default=None,
                        help="Path to aligned_50.pkl (overrides config default)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save results JSON")
    args = parser.parse_args()

    # --- Build config ---
    cfg = default_config
    cfg.apply_profile("local")
    cfg.training.task_type = args.task_type
    cfg.training.batch_size = args.batch_size
    cfg.training.num_workers = args.num_workers
    cfg.model_type = args.model_type

    if args.pkl_path:
        cfg.paths.mosei_pkl = Path(args.pkl_path)

    if cfg.model_type == "mult":
        if cfg.training.task_type == "emotion":
            cfg.mult_model.output_dim = 6
        else:
            cfg.mult_model.output_dim = 1
        cfg.mult_model.stochastic_depth_survival = cfg.training.stochastic_depth_survival

    cfg.setup()

    # --- Load checkpoint ---
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        # Try relative to project root
        ckpt_path = PROJECT_ROOT / args.checkpoint

    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Print checkpoint metadata
    print(f"  Epoch:        {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best metric:  {checkpoint.get('best_metric', 'N/A')}")
    print(f"  Valid loss:   {checkpoint.get('valid_loss', 'N/A')}")

    # --- Build model ---
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device:       {device}")

    # --- Load data ---
    dataloaders = create_dataloaders(config=cfg, pkl_path=cfg.paths.mosei_pkl)

    # --- Evaluate ---
    results = {}
    splits_to_eval = [args.split] if args.split != "all" else ["train", "valid", "test"]

    for split in splits_to_eval:
        loader = dataloaders.get(split)
        if loader is None:
            print(f"Skipping unknown split: {split}")
            continue
        print(f"\n=== {split.upper()} ===")
        avg_loss, metrics, y_pred, y_true = evaluate_split(
            model, loader, device, cfg.training.task_type
        )
        print(f"  Loss: {avg_loss:.4f}")
        if cfg.training.task_type == "emotion":
            print(f"  Mean F1:  {metrics.get('mean_f1', 0):.4f}")
            print(f"  Mean Acc: {metrics.get('mean_acc', 0):.4f}")
            print(f"  Mean MAE: {metrics.get('mean_mae', 0):.4f}")
            for emo in ["happy", "sad", "angry", "surprise", "disgust", "fear"]:
                print(f"    {emo:10s}: F1={metrics.get(f'{emo}_f1', 0):.4f}  "
                      f"Acc={metrics.get(f'{emo}_acc', 0):.4f}  "
                      f"MAE={metrics.get(f'{emo}_mae', 0):.4f}")
        else:
            print(f"  MAE:  {metrics.get('mae', 0):.4f}")
            print(f"  MSE:  {metrics.get('mse', 0):.4f}")
            print(f"  Corr: {metrics.get('corr', 0):.4f}")
            print(f"  Acc2: {metrics.get('acc2', 0):.4f}")
            print(f"  Acc5: {metrics.get('acc5', 0):.4f}")
            print(f"  Acc7: {metrics.get('acc7', 0):.4f}")
            print(f"  F1:   {metrics.get('f1', 0):.4f}")

        results[split] = {
            "loss": avg_loss,
            "metrics": metrics,
        }

    # --- Save output ---
    output = {
        "checkpoint": str(ckpt_path),
        "task_type": cfg.training.task_type,
        "model_type": cfg.model_type,
        "epoch": checkpoint.get("epoch"),
        "best_metric": checkpoint.get("best_metric"),
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x))
        print(f"\nResults saved to {out_path}")
    else:
        print(f"\n{json.dumps(output, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)}")


if __name__ == "__main__":
    main()
