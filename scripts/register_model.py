"""Register the trained RUL LSTM model in Vertex AI Model Registry.

Run from the repo root in Google Cloud Shell after Phase 3 artifacts
have been uploaded to GCS:

    python scripts/register_model.py

Prints the model resource name and model ID; save the model ID to .env
as VERTEX_MODEL_ID for use in Phase 5 (pipeline) and Phase 6 (endpoint).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Load environment variables (no third-party dotenv dependency)
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
for _line in _ENV_FILE.read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

# ---------------------------------------------------------------------------
# Vertex AI pre-built PyTorch serving container (CPU) — no local Docker needed
# GPU variant: pytorch-gpu.2-0 (higher cost)
# ---------------------------------------------------------------------------
PYTORCH_SERVING_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest"
)

MODEL_ARTIFACT_URI = f"gs://{GCS_BUCKET_NAME}/models/v1/"


def register_model() -> aiplatform.Model:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

    print(f"Registering model from: {MODEL_ARTIFACT_URI}")
    print(f"Serving container   : {PYTORCH_SERVING_CONTAINER}")

    model = aiplatform.Model.upload(
        display_name="rul-predictor-v1",
        description=(
            "PyTorch LSTM predicting turbofan engine Remaining Useful Life "
            "— NASA CMAPSS FD001 dataset. Dual-head: RUL regression + failure "
            "classification. Val RMSE=12.58, Test RMSE=15.03."
        ),
        artifact_uri=MODEL_ARTIFACT_URI,
        serving_container_image_uri=PYTORCH_SERVING_CONTAINER,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_environment_variables={
            "MODEL_NAME": "rul_predictor_v1",
            "AIP_HTTP_PORT": "8080",
        },
        labels={
            "project": "predictive-maintenance",
            "dataset": "nasa-cmapss-fd001",
            "framework": "pytorch",
            "version": "v1",
        },
    )

    print()
    print("=" * 60)
    print("Model registered successfully!")
    print(f"  resource_name : {model.resource_name}")
    print(f"  model ID      : {model.name}")
    print()
    print("Add the following to your .env file:")
    print(f"  VERTEX_MODEL_ID={model.name}")
    print("=" * 60)

    return model


if __name__ == "__main__":
    try:
        register_model()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
