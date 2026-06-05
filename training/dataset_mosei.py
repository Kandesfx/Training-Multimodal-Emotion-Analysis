from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.config_phase1 import Phase1Config, config as default_config


class MOSEIAlignedDataset(Dataset):
    def __init__(self, split: str, pkl_path: str | Path | None = None, config: Phase1Config | None = None):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.config = config or default_config
        self.split = split
        self.pkl_path = Path(pkl_path) if pkl_path is not None else self.config.paths.mosei_pkl

        payload = self._load_payload(self.pkl_path)
        split_data = payload[split]

        self.text = self._prepare_array(split_data["text"])
        self.audio = self._prepare_array(split_data["audio"], replace_inf=self.config.data.replace_inf)
        self.vision = self._prepare_array(split_data["vision"])
        self.labels = self._prepare_labels(split_data["regression_labels"])
        self.classification_labels = split_data.get("classification_labels")
        self.sample_ids = split_data.get("id", [])
        self.raw_text = split_data.get("raw_text", [])
        self.annotations = split_data.get("annotations", [])

        self._validate_shapes()

    @staticmethod
    def _load_payload(pkl_path: Path) -> dict[str, Any]:
        if not pkl_path.exists():
            raise FileNotFoundError(f"MOSEI pickle not found: {pkl_path}")
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    def _prepare_array(self, array: Any, replace_inf: bool = False) -> np.ndarray:
        arr = np.asarray(array)
        if replace_inf:
            arr = np.where(np.isfinite(arr), arr, self.config.data.audio_inf_replacement)
        if self.config.data.cast_float32:
            arr = arr.astype(np.float32, copy=False)
        return arr

    def _prepare_labels(self, labels: Any) -> np.ndarray:
        arr = np.asarray(labels)
        if self.config.data.cast_float32:
            arr = arr.astype(np.float32, copy=False)
        return arr.reshape(-1)

    def _validate_shapes(self) -> None:
        n = len(self.labels)
        expected_seq = self.config.data.sequence_length
        if self.text.shape != (n, expected_seq, self.config.model.text_input_dim):
            raise ValueError(f"Unexpected text shape: {self.text.shape}")
        if self.audio.shape != (n, expected_seq, self.config.model.audio_input_dim):
            raise ValueError(f"Unexpected audio shape: {self.audio.shape}")
        if self.vision.shape != (n, expected_seq, self.config.model.vision_input_dim):
            raise ValueError(f"Unexpected vision shape: {self.vision.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "label": torch.tensor(self.labels[index], dtype=torch.float32),
            "sample_id": self.sample_ids[index] if index < len(self.sample_ids) else str(index),
        }


def create_dataloaders(config: Phase1Config | None = None, pkl_path: str | Path | None = None) -> dict[str, DataLoader]:
    cfg = config or default_config
    datasets = {
        split: MOSEIAlignedDataset(split=split, pkl_path=pkl_path, config=cfg)
        for split in ["train", "valid", "test"]
    }
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        ),
    }
