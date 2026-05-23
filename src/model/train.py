"""Training loop for RULPredictor: early stopping, checkpointing, mixed-loss."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model.lstm import RULPredictor

_DEFAULT_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 100,
    "patience": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "model_artifacts",
    "experiment_name": "rul-predictor-v1",
    # Weight of BCELoss relative to MSELoss in combined loss
    "bce_weight": 0.3,
    # Failure threshold for generating binary classification labels
    "failure_threshold": 30,
    # Vertex AI Experiments — set True when running in GCP environment
    "use_vertex_experiments": False,
    "vertex_experiment": "rul-predictor-experiments",
    "gcp_project_id": os.environ.get("GCP_PROJECT_ID", ""),
    "gcp_region": os.environ.get("GCP_REGION", "us-central1"),
}

FAILURE_THRESHOLD = 30  # label = 1 if RUL <= this value


class Trainer:
    """Train RULPredictor with dual-head MSE + BCE loss and early stopping.

    Usage
    -----
        config = {**_DEFAULT_CONFIG, "hidden_size": 128, "num_layers": 2}
        model = RULPredictor(n_features=15)
        trainer = Trainer(model, config)
        history = trainer.train(X_train, y_train, X_val, y_val)
        trainer.save_model("model_artifacts/rul_predictor_v1.pt")

    Vertex AI Experiments (optional)
    ---------------------------------
        config["use_vertex_experiments"] = True
        config["vertex_experiment"] = "rul-predictor-experiments"
        config["gcp_project_id"] = "your-project-id"
        Run from Colab or Cloud Shell where ADC credentials are available.
    """

    def __init__(self, model: RULPredictor, config: dict | None = None) -> None:
        self.config = {**_DEFAULT_CONFIG, **(config or {})}
        self.device = torch.device(self.config["device"])
        self.model = model.to(self.device)
        self._vertex_run = None

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self._best_val_loss = float("inf")
        self._epochs_without_improvement = 0
        self._best_epoch = 0
        self.history: list[dict] = []

        Path(self.config["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict]:
        """Run the full training loop.

        Parameters
        ----------
        X_train, X_val : ndarray (n_samples, window_size, n_features)
        y_train, y_val : ndarray (n_samples,) — RUL values

        Returns
        -------
        history : list of per-epoch dicts with train_loss, val_loss, val_rmse, lr
        """
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        print(
            f"Training on {self.device} | "
            f"train={len(X_train)} | val={len(X_val)} | "
            f"batch={self.config['batch_size']} | "
            f"params={self.model.count_parameters():,}"
        )
        print("-" * 60)

        self._start_vertex_run()

        for epoch in range(1, self.config["epochs"] + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, val_rmse = self._eval_epoch(val_loader)
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                    "lr": lr,
                }
            )

            print(
                f"Epoch {epoch:3d}/{self.config['epochs']} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"rmse={val_rmse:.2f} | lr={lr:.6f} | {elapsed:.1f}s"
            )

            self._log_epoch_metrics(epoch, train_loss, val_loss, val_rmse)
            self.scheduler.step(val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
                self._save_checkpoint("best_model.pt", epoch)
            else:
                self._epochs_without_improvement += 1
                if self._epochs_without_improvement >= self.config["patience"]:
                    print(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(best epoch: {self._best_epoch}, val_loss: {self._best_val_loss:.4f})"
                    )
                    break

        print(f"\nTraining complete. Best epoch: {self._best_epoch}")
        self._load_best_checkpoint()
        return self.history

    def log_final_metrics(self, test_metrics: dict) -> None:
        """Log final test-set metrics to Vertex AI Experiments and end the run.

        Call this after Evaluator.compute_metrics() with the test results.
        Safe to call even when use_vertex_experiments=False.

        Parameters
        ----------
        test_metrics : dict with keys rmse, mae, nasa_score, within_10_pct
        """
        if not self._vertex_run:
            return
        try:
            import google.cloud.aiplatform as aip

            aip.log_metrics(
                {
                    "test_rmse": float(test_metrics.get("rmse", 0)),
                    "test_mae": float(test_metrics.get("mae", 0)),
                    "nasa_score": float(test_metrics.get("nasa_score", 0)),
                    "within_10pct": float(test_metrics.get("within_10_pct", 0)),
                    "best_epoch": float(self._best_epoch),
                }
            )
            self._vertex_run.__exit__(None, None, None)
            self._vertex_run = None
            print("Vertex AI Experiments run ended — metrics logged.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Vertex Experiments] log_final_metrics failed: {exc}")

    def save_model(self, path: str) -> None:
        """Save model state_dict + config to *path*."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config,
                "model_kwargs": {
                    "n_features": self.model.n_features,
                    "hidden_size": self.model.hidden_size,
                    "num_layers": self.model.num_layers,
                    "window_size": self.model.window_size,
                },
                "epoch": self._best_epoch,
                "val_loss": self._best_val_loss,
            },
            path,
        )
        print(f"Model saved → {path}")

    # ------------------------------------------------------------------
    # Vertex AI Experiments helpers
    # ------------------------------------------------------------------

    def _start_vertex_run(self) -> None:
        """Initialise Vertex AI Experiments and open a run context."""
        if not self.config.get("use_vertex_experiments"):
            return
        try:
            import google.cloud.aiplatform as aip

            run_name = f"run-{self.config['experiment_name']}-{int(time.time())}"
            aip.init(
                project=self.config["gcp_project_id"],
                location=self.config["gcp_region"],
                experiment=self.config["vertex_experiment"],
            )
            self._vertex_run = aip.start_run(run_name)
            self._vertex_run.__enter__()
            aip.log_params(
                {
                    "learning_rate": self.config["learning_rate"],
                    "batch_size": self.config["batch_size"],
                    "hidden_size": self.config.get("hidden_size", 128),
                    "num_layers": self.config.get("num_layers", 2),
                    "dropout": self.config.get("dropout", 0.2),
                    "window_size": self.config.get("window_size", 30),
                    "rul_cap": self.config.get("rul_cap", 125),
                    "patience": self.config["patience"],
                }
            )
            print(f"Vertex AI Experiments: run '{run_name}' started.")
        except Exception as exc:  # noqa: BLE001
            print(f"[Vertex Experiments] init failed (continuing without): {exc}")
            self._vertex_run = None

    def _log_epoch_metrics(
        self, epoch: int, train_loss: float, val_loss: float, val_rmse: float
    ) -> None:
        if not self._vertex_run:
            return
        try:
            import google.cloud.aiplatform as aip

            aip.log_time_series_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_rmse": val_rmse,
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[Vertex Experiments] epoch {epoch} log failed: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.config["batch_size"],
            shuffle=shuffle,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)

            rul_pred = outputs["rul"].squeeze(1)
            loss = self.mse_loss(rul_pred, y_batch)

            # Add classification loss: label = 1 if RUL <= FAILURE_THRESHOLD
            failure_labels = (y_batch <= FAILURE_THRESHOLD).float()
            failure_pred = outputs["failure_prob"].squeeze(1)
            bce = self.bce_loss(failure_pred, failure_labels)
            loss = loss + self.config["bce_weight"] * bce

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)

        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                rul_pred = outputs["rul"].squeeze(1)

                loss = self.mse_loss(rul_pred, y_batch)
                failure_labels = (y_batch <= FAILURE_THRESHOLD).float()
                failure_pred = outputs["failure_prob"].squeeze(1)
                bce = self.bce_loss(failure_pred, failure_labels)
                loss = loss + self.config["bce_weight"] * bce

                total_loss += loss.item() * len(X_batch)
                all_preds.append(rul_pred.cpu())
                all_targets.append(y_batch.cpu())

        val_loss = total_loss / len(loader.dataset)
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        val_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        return val_loss, val_rmse

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = Path(self.config["checkpoint_dir"]) / filename
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": self._best_val_loss,
            },
            path,
        )

    def _load_best_checkpoint(self) -> None:
        path = Path(self.config["checkpoint_dir"]) / "best_model.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")


def train_val_split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    train_df,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences by engine_id to prevent data leakage.

    Uses the last *val_frac* fraction of engine IDs for validation.
    Sequences are aligned with the training DataFrame via engine_id grouping order.
    """
    engine_ids = sorted(train_df["engine_id"].unique())
    n_val = max(1, int(len(engine_ids) * val_frac))
    val_engines = set(engine_ids[-n_val:])
    train_engines = set(engine_ids[:-n_val])

    # Build a per-row engine_id array that aligns with the flattened sequences
    engine_col = []
    for eid, group in train_df.groupby("engine_id"):
        n_windows = max(0, len(group) - 29)  # WINDOW_SIZE - 1
        engine_col.extend([eid] * n_windows)
    engine_arr = np.array(engine_col)

    assert len(engine_arr) == len(X), (
        f"Sequence count mismatch: engine_arr={len(engine_arr)}, X={len(X)}. "
        "Ensure train_df passed here is post-normalization, pre-sequence."
    )

    train_mask = np.isin(engine_arr, list(train_engines))
    val_mask = np.isin(engine_arr, list(val_engines))

    return X[train_mask], y[train_mask], X[val_mask], y[val_mask]
