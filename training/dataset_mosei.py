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
        self.emotion_labels = self._prepare_emotion_labels(split_data.get("emotion_labels"))
        self.sample_ids = split_data.get("id", [])
        self.raw_text = split_data.get("raw_text", [])
        self.annotations = split_data.get("annotations", [])
        self.task_type = self.config.training.task_type  # "sentiment" or "emotion"

        # Apply filtering for emotion mode to ignore unmatched samples (keep only matched samples)
        if self.task_type == "emotion" and "emotion_matched_mask" in split_data:
            mask = np.asarray(split_data["emotion_matched_mask"], dtype=bool)
            self.text = self.text[mask]
            self.audio = self.audio[mask]
            self.vision = self.vision[mask]
            self.labels = self.labels[mask]
            if self.classification_labels is not None:
                self.classification_labels = np.asarray(self.classification_labels)[mask]
            if self.emotion_labels is not None:
                self.emotion_labels = self.emotion_labels[mask]
            if self.sample_ids is not None and len(self.sample_ids) > 0:
                self.sample_ids = [self.sample_ids[idx] for idx, m in enumerate(mask) if m]
            if self.raw_text is not None and len(self.raw_text) > 0:
                self.raw_text = [self.raw_text[idx] for idx, m in enumerate(mask) if m]
            if self.annotations is not None and len(self.annotations) > 0:
                self.annotations = [self.annotations[idx] for idx, m in enumerate(mask) if m]

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

    def _prepare_emotion_labels(self, labels) -> np.ndarray | None:
        if labels is None:
            return None
        arr = np.asarray(labels)
        if self.config.data.cast_float32:
            arr = arr.astype(np.float32, copy=False)
        return arr  # shape: (N, 6)

    def _validate_shapes(self) -> None:
        n = len(self.labels)
        expected_seq = self.config.data.sequence_length

        # Chọn input dims dựa trên model_type
        if self.config.model_type == "mult":
            text_dim = self.config.mult_model.text_input_dim
            audio_dim = self.config.mult_model.audio_input_dim
            vision_dim = self.config.mult_model.vision_input_dim
        else:
            text_dim = self.config.model.text_input_dim
            audio_dim = self.config.model.audio_input_dim
            vision_dim = self.config.model.vision_input_dim

        if self.text.shape != (n, expected_seq, text_dim):
            raise ValueError(f"Unexpected text shape: {self.text.shape}, expected (n, {expected_seq}, {text_dim})")
        if self.audio.shape != (n, expected_seq, audio_dim):
            raise ValueError(f"Unexpected audio shape: {self.audio.shape}, expected (n, {expected_seq}, {audio_dim})")
        if self.vision.shape != (n, expected_seq, vision_dim):
            raise ValueError(f"Unexpected vision shape: {self.vision.shape}, expected (n, {expected_seq}, {vision_dim})")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Select label based on task type
        if self.task_type == "emotion" and self.emotion_labels is not None:
            label = torch.from_numpy(self.emotion_labels[index])  # (6,)
        else:
            label = torch.tensor(self.labels[index], dtype=torch.float32)  # scalar

        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "label": label,
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


# ---------------------------------------------------------------------------
# Unaligned Dataset
# ---------------------------------------------------------------------------

class MOSEIUnalignedDataset(Dataset):
    """Dataset for unaligned_50.pkl where audio/vision have 500 timesteps
    and audio_lengths/vision_lengths indicate actual (non-padded) lengths.

    Each sample returns:
        text:         (50,  768) float32
        audio:        (500,  74) float32
        vision:       (500,  35) float32
        audio_len:    scalar int64 — actual audio frames
        vision_len:   scalar int64 — actual vision frames
        label:        scalar float32
        sample_id:    str
    """

    def __init__(
        self,
        split: str,
        pkl_path: str | Path | None = None,
        config: Phase1Config | None = None,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")

        self.config = config or default_config
        self.split = split
        self.pkl_path = (
            Path(pkl_path) if pkl_path is not None
            else self.config.paths.mosei_unaligned_pkl
        )

        payload = self._load_payload(self.pkl_path)
        d = payload[split]

        self.text = self._to_f32(np.asarray(d["text"]))
        self.audio = self._to_f32(
            self._replace_inf(np.asarray(d["audio"]))
        )
        self.vision = self._to_f32(np.asarray(d["vision"]))
        self.labels = self._to_f32(np.asarray(d["regression_labels"]).reshape(-1))

        # audio_lengths / vision_lengths are lists of ints
        self.audio_lengths = np.array(
            [int(x) for x in d["audio_lengths"]], dtype=np.int64
        )
        self.vision_lengths = np.array(
            [int(x) for x in d["vision_lengths"]], dtype=np.int64
        )

        self.sample_ids: list = list(d.get("id", []))

        # Emotion labels (optional — only present after merge_emotions_to_pkl.py)
        emo_raw = d.get("emotion_labels")
        self.emotion_labels = emo_raw.astype(np.float32, copy=False) if emo_raw is not None else None
        self.task_type = self.config.training.task_type

        # Apply filtering for emotion mode to ignore unmatched samples (keep only matched samples)
        if self.task_type == "emotion" and "emotion_matched_mask" in d:
            mask = np.asarray(d["emotion_matched_mask"], dtype=bool)
            self.text = self.text[mask]
            self.audio = self.audio[mask]
            self.vision = self.vision[mask]
            self.labels = self.labels[mask]
            self.audio_lengths = self.audio_lengths[mask]
            self.vision_lengths = self.vision_lengths[mask]
            if self.emotion_labels is not None:
                self.emotion_labels = self.emotion_labels[mask]
            if self.sample_ids is not None and len(self.sample_ids) > 0:
                self.sample_ids = [self.sample_ids[idx] for idx, m in enumerate(mask) if m]

        self._validate_shapes()

    # ------------------------------------------------------------------
    @staticmethod
    def _load_payload(pkl_path: Path) -> dict[str, Any]:
        if not pkl_path.exists():
            raise FileNotFoundError(f"MOSEI unaligned pickle not found: {pkl_path}")
        with pkl_path.open("rb") as f:
            return pickle.load(f)

    def _to_f32(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32, copy=False)

    def _replace_inf(self, arr: np.ndarray) -> np.ndarray:
        if self.config.data.replace_inf:
            arr = np.where(np.isfinite(arr), arr, self.config.data.audio_inf_replacement)
        return arr

    def _validate_shapes(self) -> None:
        n = len(self.labels)
        text_seq = self.config.data.sequence_length          # 50
        av_seq = self.config.data.audio_vision_seq_len        # 500
        text_dim = self.config.mult_model.text_input_dim     # 768
        audio_dim = self.config.mult_model.audio_input_dim   # 74
        vision_dim = self.config.mult_model.vision_input_dim # 35

        assert self.text.shape == (n, text_seq, text_dim), (
            f"text shape mismatch: {self.text.shape} vs ({n},{text_seq},{text_dim})"
        )
        assert self.audio.shape == (n, av_seq, audio_dim), (
            f"audio shape mismatch: {self.audio.shape} vs ({n},{av_seq},{audio_dim})"
        )
        assert self.vision.shape == (n, av_seq, vision_dim), (
            f"vision shape mismatch: {self.vision.shape} vs ({n},{av_seq},{vision_dim})"
        )
        assert len(self.audio_lengths) == n, "audio_lengths length mismatch"
        assert len(self.vision_lengths) == n, "vision_lengths length mismatch"

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        # Select label based on task type
        if self.task_type == "emotion" and self.emotion_labels is not None:
            label = torch.from_numpy(self.emotion_labels[index])  # (6,)
        else:
            label = torch.tensor(self.labels[index], dtype=torch.float32)  # scalar

        return {
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "audio_len": torch.tensor(self.audio_lengths[index], dtype=torch.long),
            "vision_len": torch.tensor(self.vision_lengths[index], dtype=torch.long),
            "label": label,
            "sample_id": self.sample_ids[index] if index < len(self.sample_ids) else str(index),
        }


def create_unaligned_dataloaders(
    config: Phase1Config | None = None,
    pkl_path: str | Path | None = None,
) -> dict[str, DataLoader]:
    """Create DataLoaders for unaligned_50.pkl.

    Uses a standard collate (lengths are scalars so they stack fine).
    Recommended: use batch_size=16 to accommodate 500-timestep sequences.
    """
    cfg = config or default_config
    datasets = {
        split: MOSEIUnalignedDataset(split=split, pkl_path=pkl_path, config=cfg)
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
