from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def _safe_corrcoef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _binary_labels(values: np.ndarray) -> np.ndarray:
    return (values >= 0).astype(int)


def _clip_and_round(values: np.ndarray, low: int, high: int) -> np.ndarray:
    return np.rint(np.clip(values, low, high)).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    corr = _safe_corrcoef(y_true, y_pred)

    true_bin = _binary_labels(y_true)
    pred_bin = _binary_labels(y_pred)
    acc2 = float(accuracy_score(true_bin, pred_bin))
    f1 = float(f1_score(true_bin, pred_bin, average="weighted"))

    true_acc7 = _clip_and_round(y_true, -3, 3)
    pred_acc7 = _clip_and_round(y_pred, -3, 3)
    acc7 = float(accuracy_score(true_acc7, pred_acc7))

    true_acc5 = _clip_and_round(y_true, -2, 2)
    pred_acc5 = _clip_and_round(y_pred, -2, 2)
    acc5 = float(accuracy_score(true_acc5, pred_acc5))

    return {
        "mae": mae,
        "mse": mse,
        "corr": corr,
        "acc2": acc2,
        "acc5": acc5,
        "acc7": acc7,
        "f1": f1,
    }


def metrics_to_row(split: str, epoch: int, loss: float, metrics: dict[str, float]) -> dict[str, Any]:
    row = {"split": split, "epoch": epoch, "loss": float(loss)}
    row.update({key: float(value) for key, value in metrics.items()})
    return row
