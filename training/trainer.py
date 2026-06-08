from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
)
from tqdm import tqdm

from training.config_phase1 import Phase1Config
from training.evaluator import compute_metrics, metrics_to_row
from training.evaluator_emotion import compute_emotion_metrics, emotion_metrics_to_row


class _CombinedMSEL1Loss(nn.Module):
    """Combined MSE + L1 loss for sentiment regression.

    Directly optimizes both squared error and absolute error.
    L1 component aligns training signal with the evaluation metric (MAE).

    loss = (1 - l1_weight) * MSE + l1_weight * L1
    """

    def __init__(self, l1_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return (1 - self.l1_weight) * self.mse(pred, target) + self.l1_weight * self.l1(pred, target)




class Phase1Trainer:
    def __init__(self, model: nn.Module, config: Phase1Config, device: torch.device | None = None):
        self.model = model
        self.config = config
        self.config.setup()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = self._build_criterion()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        self.scheduler = self._build_scheduler()
        self.scheduler_type = self.config.training.scheduler_type
        self.task_type = self.config.training.task_type  # "sentiment" or "emotion"
        if self.task_type == "emotion":
            self.metric_for_best = "mean_f1"
            self.maximize_metric = True
        else:
            self.metric_for_best = self.config.training.metric_for_best
            self.maximize_metric = self.config.training.maximize_metric
        self.use_amp = self.config.training.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history_path = self.config.paths.logs_dir / "history.csv"
        self.summary_path = self.config.paths.outputs_dir / "summary.json"
        self.best_checkpoint_path = self.config.paths.checkpoints_dir / self.config.training.checkpoint_name
        self.last_checkpoint_path = self.config.paths.checkpoints_dir / self.config.training.last_checkpoint_name

        if self.config.wandb.enable:
            import wandb
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                config=asdict(self.config),
                name=f"phase1_{self.config.model_type}_{self.config.runtime.profile}"
            )
            wandb.watch(self.model, log="all")

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_criterion(self) -> nn.Module:
        """Build loss function based on config.training.loss_type."""
        loss_type = self.config.training.loss_type.lower().strip()
        if loss_type == "mse":
            return nn.MSELoss()
        if loss_type == "mse_l1":
            return _CombinedMSEL1Loss(l1_weight=self.config.training.l1_weight)
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"Unsupported loss_type: {loss_type!r}. Use 'mse', 'mse_l1', or 'bce'.")

    def _build_scheduler(self):
        """Build LR scheduler based on config.training.scheduler_type."""
        sched_type = self.config.training.scheduler_type.lower().strip()
        if sched_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
            )
        if sched_type == "cosine_warmup":
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,           # start at 1% of peak LR
                total_iters=self.config.training.warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.training.num_epochs - self.config.training.warmup_epochs),
                eta_min=self.config.training.min_lr,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.config.training.warmup_epochs],
            )
        raise ValueError(f"Unsupported scheduler_type: {sched_type!r}. Use 'plateau' or 'cosine_warmup'.")

    def _upload_to_gcs(self, local_path: Path) -> None:
        if not self.config.runtime.use_gcs:
            return
        
        import subprocess
        bucket = self.config.runtime.gcs_bucket
        
        if "checkpoints" in str(local_path):
            folder = "checkpoints/phase1"
        elif "logs" in str(local_path):
            folder = "logs/phase1"
        elif "outputs" in str(local_path):
            folder = "outputs/phase1"
        else:
            folder = "artifacts/phase1"
            
        gcs_dest = f"gs://{bucket}/{folder}/{local_path.name}"
        try:
            subprocess.run(["gsutil", "cp", str(local_path), gcs_dest], check=True, capture_output=True)
            print(f"Successfully uploaded {local_path.name} to {gcs_dest}")
        except Exception as e:
            print(f"Failed to upload {local_path} to GCS: {e}")

    def fit(self, train_loader, valid_loader) -> dict[str, Any]:
        self.set_seed(self.config.training.seed)

        start_epoch = 1
        best_metric = float("inf") if not self.maximize_metric else float("-inf")
        epochs_without_improvement = 0
        history_rows = self._load_existing_history_count() if self.config.training.resume_from_checkpoint else 0
        best_state: dict[str, Any] | None = None
        resumed_from = None
        last_completed_epoch = 0

        if self.config.training.resume_from_checkpoint:
            checkpoint_state, resumed_from = self._load_resume_checkpoint()
            if checkpoint_state is not None:
                last_completed_epoch = int(checkpoint_state.get("epoch", 0))
                start_epoch = last_completed_epoch + 1
                best_metric = float(checkpoint_state.get("best_metric", best_metric))
                epochs_without_improvement = int(checkpoint_state.get("epochs_without_improvement", 0))
                best_state = self._load_best_checkpoint_state(optional=True) or checkpoint_state
            else:
                self._reset_history_for_fresh_run()
        else:
            self._reset_history_for_fresh_run()

        for epoch in range(start_epoch, self.config.training.num_epochs + 1):
            train_loss = self._run_epoch(train_loader, training=True, epoch=epoch)
            train_eval_loss, train_metrics = self.evaluate(train_loader, split="train", epoch=epoch)
            valid_loss, valid_metrics = self.evaluate(valid_loader, split="valid", epoch=epoch)
            # Scheduler step — plateau uses metric, cosine_warmup uses epoch count
            if self.scheduler_type == "plateau":
                self.scheduler.step(valid_loss)
            else:
                self.scheduler.step()

            current_metric = valid_metrics[self.metric_for_best]
            improved = current_metric > best_metric if self.maximize_metric else current_metric < best_metric
            if improved:
                best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if self.config.wandb.enable:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/eval_loss": train_eval_loss,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    "val/loss": valid_loss,
                    **{f"val/{k}": v for k, v in valid_metrics.items()},
                    "best_metric": best_metric,
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                })

            checkpoint_state = self._build_checkpoint_state(
                epoch=epoch,
                train_loss=train_loss,
                valid_loss=valid_loss,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                best_metric=best_metric,
                epochs_without_improvement=epochs_without_improvement,
            )
            torch.save(checkpoint_state, self.last_checkpoint_path)
            self._upload_to_gcs(self.last_checkpoint_path)
            last_completed_epoch = epoch

            if self.task_type == "emotion":
                train_row = emotion_metrics_to_row("train", epoch, train_eval_loss, train_metrics)
                train_row["train_step_loss"] = float(train_loss)
                valid_row = emotion_metrics_to_row("valid", epoch, valid_loss, valid_metrics)
            else:
                train_row = metrics_to_row("train", epoch, train_eval_loss, train_metrics)
                train_row["train_step_loss"] = float(train_loss)
                valid_row = metrics_to_row("valid", epoch, valid_loss, valid_metrics)
            self._append_history([train_row, valid_row])
            self._upload_to_gcs(self.history_path)
            history_rows += 2

            if improved:
                best_state = checkpoint_state
                torch.save(checkpoint_state, self.best_checkpoint_path)
                self._upload_to_gcs(self.best_checkpoint_path)

            if epochs_without_improvement >= self.config.training.patience:
                break

        if best_state is None:
            raise RuntimeError("Training ended without producing a best checkpoint.")

        self.model.load_state_dict(best_state["model_state_dict"])
        summary = {
            "best_epoch": best_state["epoch"],
            "best_metric": float(best_state["best_metric"]),
            "history_rows": history_rows,
            "best_checkpoint_path": str(self.best_checkpoint_path),
            "last_checkpoint_path": str(self.last_checkpoint_path),
            "resumed": start_epoch > 1,
            "resume_checkpoint_type": resumed_from,
            "start_epoch": start_epoch,
            "last_completed_epoch": last_completed_epoch,
            "end_epoch": best_state["epoch"],
            "device": str(self.device),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._upload_to_gcs(self.summary_path)

        if self.config.wandb.enable:
            import wandb
            wandb.finish()

        return summary

    def load_best_checkpoint(self) -> dict[str, Any]:
        state = self._load_checkpoint_state(self.best_checkpoint_path)
        self.model.load_state_dict(state["model_state_dict"])
        return state

    def load_last_checkpoint(self) -> dict[str, Any]:
        state = self._load_checkpoint_state(self.last_checkpoint_path)
        self.model.load_state_dict(state["model_state_dict"])
        return state

    def evaluate(self, data_loader, split: str = "valid", epoch: int = 0) -> tuple[float, dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        with torch.no_grad():
            for batch in data_loader:
                text = batch["text"].to(self.device, non_blocking=True)
                audio = batch["audio"].to(self.device, non_blocking=True)
                vision = batch["vision"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                # Unaligned batches carry length tensors; aligned batches do not.
                audio_lengths = batch["audio_len"].to(self.device, non_blocking=True) if "audio_len" in batch else None
                vision_lengths = batch["vision_len"].to(self.device, non_blocking=True) if "vision_len" in batch else None

                preds = self.model(
                    text=text, audio=audio, vision=vision,
                    audio_lengths=audio_lengths, vision_lengths=vision_lengths,
                )
                loss = self.criterion(preds, labels)
                total_loss += loss.item() * labels.size(0)
                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / len(data_loader.dataset)

        # Dispatch to correct evaluator based on task type
        if self.task_type == "emotion":
            metrics = compute_emotion_metrics(y_true, y_pred)
        else:
            metrics = compute_metrics(y_true, y_pred)

        return avg_loss, metrics

    def evaluate_and_save(self, data_loader, split: str = "test", epoch: int = 0) -> dict[str, Any]:
        loss, metrics = self.evaluate(data_loader, split=split, epoch=epoch)
        if self.task_type == "emotion":
            row = emotion_metrics_to_row(split, epoch, loss, metrics)
        else:
            row = metrics_to_row(split, epoch, loss, metrics)
        self._append_history([row])
        return row

    def _run_epoch(self, data_loader, training: bool, epoch: int) -> float:
        self.model.train(training)
        total_loss = 0.0
        progress = tqdm(data_loader, desc=f"Epoch {epoch} {'train' if training else 'eval'}", leave=False)
        accum_steps = self.config.training.gradient_accumulation_steps

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            text = batch["text"].to(self.device, non_blocking=True)
            audio = batch["audio"].to(self.device, non_blocking=True)
            vision = batch["vision"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Unaligned batches carry length tensors; aligned batches do not.
            audio_lengths = batch["audio_len"].to(self.device, non_blocking=True) if "audio_len" in batch else None
            vision_lengths = batch["vision_len"].to(self.device, non_blocking=True) if "vision_len" in batch else None

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                preds = self.model(
                    text=text, audio=audio, vision=vision,
                    audio_lengths=audio_lengths, vision_lengths=vision_lengths,
                )
                raw_loss = self.criterion(preds, labels)
                loss = raw_loss / accum_steps

            if training:
                # Guard: skip batch if loss is NaN (prevents poisoning optimizer state)
                if not torch.isfinite(raw_loss):
                    print(f"\n  [WARNING] Non-finite loss ({raw_loss.item()}) at step {step} — skipping batch")
                    continue

                self.scaler.scale(loss).backward()

                if step % accum_steps == 0 or step == len(data_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

            total_loss += raw_loss.item() * labels.size(0)
            if step % self.config.training.log_interval == 0 or step == len(data_loader):
                progress.set_postfix(loss=f"{raw_loss.item():.4f}")

        return total_loss / len(data_loader.dataset)

    def _append_history(self, rows: list[dict[str, Any]]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.history_path.exists()
        if self.task_type == "emotion":
            fieldnames = [
                "split", "epoch", "loss", "train_step_loss",
                "mean_f1", "mean_acc", "mean_mae",
                "happy_f1", "sad_f1", "angry_f1",
                "surprise_f1", "disgust_f1", "fear_f1",
            ]
        else:
            fieldnames = ["split", "epoch", "loss", "mae", "mse", "corr", "acc2", "acc5", "acc7", "f1", "train_step_loss"]
        with self.history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in rows:
                normalized = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(normalized)

    def _build_checkpoint_state(
        self,
        epoch: int,
        train_loss: float,
        valid_loss: float,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float],
        best_metric: float,
        epochs_without_improvement: int,
    ) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
            "best_metric": float(best_metric),
            "epochs_without_improvement": epochs_without_improvement,
            "config": asdict(self.config),
        }

    def _load_resume_checkpoint(self) -> tuple[dict[str, Any] | None, str | None]:
        checkpoint_type = self.config.training.resume_checkpoint_type.lower().strip()

        if checkpoint_type == "last":
            if self.last_checkpoint_path.exists():
                return self._restore_training_state(self.last_checkpoint_path), "last"
            if self.best_checkpoint_path.exists():
                return self._restore_training_state(self.best_checkpoint_path), "best_fallback"
            return None, None

        if checkpoint_type == "best":
            if self.best_checkpoint_path.exists():
                return self._restore_training_state(self.best_checkpoint_path), "best"
            return None, None

        raise ValueError(f"Unsupported resume checkpoint type: {self.config.training.resume_checkpoint_type}")

    def _restore_training_state(self, checkpoint_path: Path) -> dict[str, Any]:
        state = self._load_checkpoint_state(checkpoint_path)
        self.model.load_state_dict(state["model_state_dict"])

        optimizer_state = state.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = state.get("scheduler_state_dict")
        if scheduler_state is not None:
            self.scheduler.load_state_dict(scheduler_state)

        scaler_state = state.get("scaler_state_dict")
        if scaler_state is not None and self.use_amp:
            self.scaler.load_state_dict(scaler_state)

        return state

    def _load_best_checkpoint_state(self, optional: bool = False) -> dict[str, Any] | None:
        if not self.best_checkpoint_path.exists():
            if optional:
                return None
            raise FileNotFoundError(f"Checkpoint not found: {self.best_checkpoint_path}")
        return self._load_checkpoint_state(self.best_checkpoint_path)

    def _load_checkpoint_state(self, checkpoint_path: Path) -> dict[str, Any]:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)

    def _load_existing_history_count(self) -> int:
        if not self.history_path.exists():
            return 0

        with self.history_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    def _reset_history_for_fresh_run(self) -> None:
        if self.history_path.exists():
            self.history_path.unlink(missing_ok=True)
