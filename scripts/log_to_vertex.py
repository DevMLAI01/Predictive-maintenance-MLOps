"""Log Phase 9 experiment results to Vertex AI Experiments.

Reads experiment_summary.json and creates 3 experiment runs in
Vertex AI — no retraining required.

Run in Cloud Shell:
    python scripts/log_to_vertex.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "groovy-rope-496901-d2")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
EXPERIMENT_NAME = "rul-predictor-experiments"
SUMMARY_PATH = _ROOT / "model_artifacts" / "experiment_summary.json"

RUNS = [
    {
        "run_name": "run-1-baseline",
        "params": {
            "hidden_size": 64,
            "num_layers": 1,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "batch_size": 256,
            "window_size": 30,
            "rul_cap": 125,
        },
        "metrics": {
            "test_rmse": 15.26,
            "nasa_score": 496.9,
            "within_10pct": 40.0,
        },
    },
    {
        "run_name": "run-2-deeper",
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.001,
            "dropout": 0.2,
            "batch_size": 256,
            "window_size": 30,
            "rul_cap": 125,
        },
        "metrics": {
            "test_rmse": 15.59,
            "nasa_score": 477.5,
            "within_10pct": 37.0,
        },
    },
    {
        "run_name": "run-3-optimized",
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "learning_rate": 0.0005,
            "dropout": 0.3,
            "batch_size": 256,
            "window_size": 30,
            "rul_cap": 125,
        },
        "metrics": {
            "test_rmse": 15.25,
            "nasa_score": 427.6,
            "within_10pct": 38.0,
        },
    },
]


def main() -> None:
    from google.cloud import aiplatform

    print(f"Project : {GCP_PROJECT_ID}")
    print(f"Region  : {GCP_REGION}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print()

    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=GCP_REGION,
        experiment=EXPERIMENT_NAME,
    )
    print(f"Experiment '{EXPERIMENT_NAME}' initialised.")

    for run in RUNS:
        print(f"\nLogging {run['run_name']} ...")
        aiplatform.start_run(run["run_name"])
        aiplatform.log_params(run["params"])
        aiplatform.log_metrics(run["metrics"])
        aiplatform.end_run()
        print(f"  params : {run['params']}")
        print(f"  metrics: {run['metrics']}")
        print(f"  ✓ {run['run_name']} logged and ended.")

    print("\nAll 3 runs logged to Vertex AI Experiments.")
    print(
        f"View at: https://console.cloud.google.com/vertex-ai/experiments"
        f"?project={GCP_PROJECT_ID}"
    )


if __name__ == "__main__":
    main()
