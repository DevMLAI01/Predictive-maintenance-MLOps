"""Phase 9 — Vertex AI Experiments: run 3 hyperparameter variants and log results.

Run this script in Google Colab or Cloud Shell where:
  - GCP credentials are available (Application Default Credentials)
  - Processed data exists at data/processed/ or GCS
  - Environment variables GCP_PROJECT_ID and GCP_REGION are set

Usage
-----
    python scripts/run_experiments.py
    python scripts/run_experiments.py --data-dir gs://predictive-maintenance-artifacts/data/processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add repo root to path so "src.*" is importable when run as a script
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.model.lstm import RULPredictor
from src.model.train import Trainer, train_val_split_by_engine
from src.model.evaluate import Evaluator

# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENT_RUNS = [
    {
        "run_id": "run-1-baseline",
        "label": "Baseline (64-1-0.001)",
        "overrides": {
            "hidden_size": 64,
            "num_layers": 1,
            "learning_rate": 0.001,
            "dropout": 0.2,
        },
    },
    {
        "run_id": "run-2-deeper",
        "label": "Deeper (128-2-0.001)",
        "overrides": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.001,
            "dropout": 0.2,
        },
    },
    {
        "run_id": "run-3-optimized",
        "label": "Optimized (128-2-0.0005-d0.3)",
        "overrides": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.0005,
            "dropout": 0.3,
        },
    },
]


def load_processed_data(data_dir: str) -> tuple:
    """Load processed numpy arrays from local or GCS path."""
    data_path = Path(data_dir)

    if str(data_dir).startswith("gs://"):
        import subprocess

        local_tmp = Path("/tmp/processed_data")
        local_tmp.mkdir(exist_ok=True)
        print(f"Downloading processed data from {data_dir} ...")
        subprocess.run(
            ["gsutil", "-m", "cp", f"{data_dir}/*.npy", str(local_tmp)],
            check=True,
        )
        data_path = local_tmp

    X_train = np.load(data_path / "X_train.npy")
    y_train = np.load(data_path / "y_train.npy")
    X_test = np.load(data_path / "X_test.npy")
    y_test = np.load(data_path / "y_test.npy")

    print(
        f"Data loaded — X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"X_test: {X_test.shape}, y_test: {y_test.shape}"
    )
    return X_train, y_train, X_test, y_test


def run_single_experiment(
    run_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_config: dict,
) -> dict:
    """Train one model variant and return its metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {run_cfg['label']}")
    print(f"{'=' * 60}")

    n_features = X_train.shape[2]
    model = RULPredictor(
        n_features=n_features,
        hidden_size=run_cfg["overrides"].get("hidden_size", 128),
        num_layers=run_cfg["overrides"].get("num_layers", 2),
        dropout=run_cfg["overrides"].get("dropout", 0.2),
    )

    config = {
        **base_config,
        **run_cfg["overrides"],
        "experiment_name": run_cfg["run_id"],
    }

    trainer = Trainer(model, config)
    trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate on held-out test set
    evaluator = Evaluator()
    import torch

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_t)
        y_pred = outputs["rul"].squeeze(1).cpu().numpy()

    metrics = evaluator.compute_metrics(y_test, y_pred)

    # Log final test metrics to Vertex AI Experiments
    trainer.log_final_metrics(metrics)

    # Save checkpoint for this run
    ckpt_path = f"model_artifacts/{run_cfg['run_id']}.pt"
    trainer.save_model(ckpt_path)

    result = {
        "run_id": run_cfg["run_id"],
        "label": run_cfg["label"],
        "hyperparams": {
            "hidden_size": run_cfg["overrides"].get("hidden_size"),
            "num_layers": run_cfg["overrides"].get("num_layers"),
            "learning_rate": run_cfg["overrides"].get("learning_rate"),
            "dropout": run_cfg["overrides"].get("dropout"),
        },
        "metrics": {
            "test_rmse": round(float(metrics["rmse"]), 2),
            "test_mae": round(float(metrics["mae"]), 2),
            "nasa_score": round(float(metrics["nasa_score"]), 1),
            "within_10pct": round(float(metrics["within_10_pct"]), 1),
        },
        "best_checkpoint": ckpt_path,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    print(
        f"\n  RMSE: {result['metrics']['test_rmse']:.2f} | "
        f"NASA Score: {result['metrics']['nasa_score']:.1f} | "
        f"Within 10%: {result['metrics']['within_10pct']:.1f}%"
    )
    return result


def identify_best_run(results: list[dict]) -> dict:
    return min(results, key=lambda r: r["metrics"]["test_rmse"])


def save_summary(results: list[dict], output_path: str) -> None:
    best = identify_best_run(results)
    summary = {
        "experiment": "rul-predictor-experiments",
        "runs": results,
        "best_run": best["run_id"],
        "best_rmse": best["metrics"]["test_rmse"],
        "best_nasa_score": best["metrics"]["nasa_score"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nExperiment summary saved → {output_path}")

    # Print comparison table
    print("\n" + "=" * 72)
    print(f"  {'Run':<35} {'RMSE':>7} {'NASA':>8} {'Within10%':>10}")
    print("-" * 72)
    for r in results:
        marker = " ← best" if r["run_id"] == best["run_id"] else ""
        print(
            f"  {r['label']:<35} "
            f"{r['metrics']['test_rmse']:>7.2f} "
            f"{r['metrics']['nasa_score']:>8.1f} "
            f"{r['metrics']['within_10pct']:>9.1f}%"
            f"{marker}"
        )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3 Vertex AI Experiment variants")
    parser.add_argument(
        "--data-dir",
        default="data/processed",
        help="Local path or gs:// URI to processed .npy files",
    )
    parser.add_argument(
        "--output",
        default="model_artifacts/experiment_summary.json",
        help="Path to save experiment summary JSON",
    )
    parser.add_argument(
        "--vertex-experiments",
        action="store_true",
        help="Enable Vertex AI Experiments logging (requires GCP credentials)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Max training epochs per run (reduce for quick smoke-test)",
    )
    args = parser.parse_args()

    X_train_full, y_train_full, X_test, y_test = load_processed_data(args.data_dir)

    # Need the original train_df for engine-aware split — load from processed parquet
    # if available, otherwise fall back to percentage split by row count
    train_df_path = Path(args.data_dir.replace("gs://", "/tmp/")) / "train_df.parquet"
    if train_df_path.exists():
        import pandas as pd

        train_df = pd.read_parquet(train_df_path)
        X_train, y_train, X_val, y_val = train_val_split_by_engine(
            X_train_full, y_train_full, train_df
        )
    else:
        # Simple 80/20 split when train_df isn't available
        n = len(X_train_full)
        split = int(n * 0.8)
        X_train, y_train = X_train_full[:split], y_train_full[:split]
        X_val, y_val = X_train_full[split:], y_train_full[split:]
        print("Note: using row-based 80/20 split (train_df.parquet not found).")

    base_config = {
        "epochs": args.epochs,
        "batch_size": 256,
        "patience": 15,
        "bce_weight": 0.3,
        "failure_threshold": 30,
        "window_size": 30,
        "rul_cap": 125,
        "use_vertex_experiments": args.vertex_experiments,
        "vertex_experiment": "rul-predictor-experiments",
        "gcp_project_id": os.environ.get("GCP_PROJECT_ID", ""),
        "gcp_region": os.environ.get("GCP_REGION", "us-central1"),
    }

    results = []
    for run_cfg in EXPERIMENT_RUNS:
        result = run_single_experiment(
            run_cfg,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            base_config,
        )
        results.append(result)

    save_summary(results, args.output)

    best = identify_best_run(results)
    print(f"\nBest run: {best['label']}")
    print(
        f"  RMSE {best['metrics']['test_rmse']:.2f} | NASA Score {best['metrics']['nasa_score']:.1f}"
    )
    print("\nUpdate WORKING_NOTES.md Experiment Results table with the values above.")


if __name__ == "__main__":
    main()
