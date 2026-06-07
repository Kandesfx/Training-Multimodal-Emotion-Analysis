"""Emotion-specific evaluation metrics for 6-emotion multi-label classification.

Emotions: [happy, sad, angry, surprise, disgust, fear]
Labels: intensity scale 0-3, binarized at threshold for classification metrics.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


EMOTION_NAMES = ["happy", "sad", "angry", "surprise", "disgust", "fear"]

# Threshold: intensity >= THRESHOLD means "emotion is present"
DEFAULT_THRESHOLD = 0.5


def compute_emotion_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute emotion classification metrics.

    Args:
        y_true: (N, 6) float32 — ground truth emotion intensities (0-3)
        y_pred: (N, 6) float32 — predicted logits (raw, before sigmoid)
        threshold: binarization threshold for intensity labels

    Returns:
        Dict with per-emotion and aggregate metrics.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 6)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 6)

    assert y_true.shape[1] == 6 and y_pred.shape[1] == 6, (
        f"Expected 6 emotions, got true={y_true.shape}, pred={y_pred.shape}"
    )

    # Guard: filter NaN/Inf
    valid_mask = np.isfinite(y_pred).all(axis=1) & np.isfinite(y_true).all(axis=1)
    if not valid_mask.all():
        n_invalid = int((~valid_mask).sum())
        warnings.warn(
            f"compute_emotion_metrics: {n_invalid}/{len(y_pred)} NaN/Inf predictions filtered.",
            RuntimeWarning, stacklevel=2,
        )
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        return {f"{emo}_f1": 0.0 for emo in EMOTION_NAMES} | {"mean_f1": 0.0, "mean_acc": 0.0}

    # Binarize ground truth: intensity >= threshold -> 1 (emotion present)
    y_true_bin = (y_true >= threshold).astype(int)

    # Binarize predictions: apply sigmoid then threshold at 0.5
    y_pred_prob = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20, 20)))  # sigmoid
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    metrics = {}
    f1_scores = []
    acc_scores = []

    for i, emo in enumerate(EMOTION_NAMES):
        true_col = y_true_bin[:, i]
        pred_col = y_pred_bin[:, i]

        # Per-emotion F1 (binary)
        emo_f1 = float(f1_score(true_col, pred_col, average="binary", zero_division=0))
        emo_acc = float(accuracy_score(true_col, pred_col))

        metrics[f"{emo}_f1"] = emo_f1
        metrics[f"{emo}_acc"] = emo_acc

        f1_scores.append(emo_f1)
        acc_scores.append(emo_acc)

    # Aggregate metrics
    metrics["mean_f1"] = float(np.mean(f1_scores))
    metrics["mean_acc"] = float(np.mean(acc_scores))

    # Also compute MAE per emotion (regression quality)
    mae_per_emo = np.mean(np.abs(y_true - y_pred_prob * 3.0), axis=0)  # scale sigmoid back to 0-3
    for i, emo in enumerate(EMOTION_NAMES):
        metrics[f"{emo}_mae"] = float(mae_per_emo[i])
    metrics["mean_mae"] = float(np.mean(mae_per_emo))

    return metrics


def emotion_metrics_to_row(
    split: str, epoch: int, loss: float, metrics: dict[str, float]
) -> dict[str, Any]:
    """Format emotion metrics as a CSV-compatible row."""
    row = {"split": split, "epoch": epoch, "loss": float(loss)}
    # Only include key metrics to keep CSV manageable
    for key in ["mean_f1", "mean_acc", "mean_mae",
                 "happy_f1", "sad_f1", "angry_f1",
                 "surprise_f1", "disgust_f1", "fear_f1"]:
        if key in metrics:
            row[key] = float(metrics[key])
    return row
