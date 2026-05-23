"""Evaluation metrics for RUL prediction: RMSE, MAE, NASA score, plots, report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# FD001 benchmark targets from literature
BENCHMARK = {
    "rmse_good": 15.0,
    "rmse_excellent": 13.0,
    "nasa_score_good": 300.0,
    "nasa_score_excellent": 200.0,
}


class Evaluator:
    """Compute and report RUL prediction quality metrics."""

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Compute RMSE, MAE, NASA score, and within-10% accuracy.

        NASA Scoring Function (asymmetric — late predictions penalised more):
            d = y_pred - y_true
            s_i = exp(-d/13) - 1   if d < 0   (early prediction)
            s_i = exp( d/10) - 1   if d >= 0  (late prediction)
            score = sum(s_i)
        Lower score is better.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))

        d = y_pred - y_true
        nasa_score = float(
            np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
        )

        tol = np.abs(y_true) * 0.10
        tol = np.maximum(tol, 1.0)  # avoid zero-tolerance at RUL=0
        within_10_pct = float(np.mean(np.abs(y_pred - y_true) <= tol) * 100)

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "nasa_score": round(nasa_score, 2),
            "within_10_pct": round(within_10_pct, 2),
        }

    def benchmark_grade(self, metrics: dict[str, float]) -> str:
        """Return 'excellent', 'good', or 'below_target' based on RMSE."""
        rmse = metrics["rmse"]
        if rmse < BENCHMARK["rmse_excellent"]:
            return "excellent"
        if rmse < BENCHMARK["rmse_good"]:
            return "good"
        return "below_target"

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = "model_artifacts/evaluation_plots",
        sample_engines: int = 5,
    ) -> None:
        """Generate and save two evaluation plots.

        1. Scatter: predicted vs actual RUL for all test engines.
        2. Line: RUL over engine index for a random sample.
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # --- Scatter plot ---
        fig, ax = plt.subplots(figsize=(7, 7))
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
        ax.scatter(y_true, y_pred, alpha=0.6, s=25, edgecolors="none")
        ax.set_xlabel("Actual RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title("Predicted vs Actual RUL — NASA CMAPSS FD001")
        ax.legend()
        ax.grid(True, alpha=0.3)
        scatter_path = save_dir / "scatter_pred_vs_actual.png"
        fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Scatter plot → {scatter_path}")

        # --- Line plot: sample engines ---
        n = min(sample_engines, len(y_true))
        idx = np.linspace(0, len(y_true) - 1, n, dtype=int)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_true[idx], "o-", label="Actual RUL", color="steelblue")
        ax.plot(y_pred[idx], "s--", label="Predicted RUL", color="tomato")
        ax.set_xlabel("Engine index (sampled)")
        ax.set_ylabel("RUL (cycles)")
        ax.set_title(f"RUL Predictions for {n} Sample Engines")
        ax.legend()
        ax.grid(True, alpha=0.3)
        line_path = save_dir / "line_sample_engines.png"
        fig.savefig(line_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Line plot     → {line_path}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        metrics: dict[str, float],
        save_path: str = "model_artifacts/evaluation_report.json",
    ) -> None:
        """Save metrics + benchmark comparison to JSON and print a summary table."""
        grade = self.benchmark_grade(metrics)
        report = {
            **metrics,
            "grade": grade,
            "benchmark": BENCHMARK,
        }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

        rmse_target = (
            f"✅ excellent (< {BENCHMARK['rmse_excellent']})"
            if grade == "excellent"
            else (
                f"✅ good (< {BENCHMARK['rmse_good']})"
                if grade == "good"
                else f"⚠️  below target (target < {BENCHMARK['rmse_good']})"
            )
        )

        print("\n" + "=" * 50)
        print("  Evaluation Results — NASA CMAPSS FD001")
        print("=" * 50)
        print(f"  RMSE          : {metrics['rmse']:.2f} cycles  {rmse_target}")
        print(f"  MAE           : {metrics['mae']:.2f} cycles")
        print(
            f"  NASA Score    : {metrics['nasa_score']:.1f}  "
            f"(target < {BENCHMARK['nasa_score_good']})"
        )
        print(f"  Within 10%    : {metrics['within_10_pct']:.1f}%")
        print(f"  Grade         : {grade.upper()}")
        print("=" * 50)
        print(f"  Report saved  → {save_path}\n")
