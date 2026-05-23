"""Register a new Vertex AI Model with the custom container and deploy it.

Run after scripts/build_and_push_container.sh has pushed the container image.

Steps performed:
  1. Upload scaler_params.npz to GCS (if not already there)
  2. Register a new Vertex AI Model resource pointing at the custom container
  3. Deploy the model to the existing endpoint

Usage:
    python scripts/register_and_deploy_custom.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.cloud import aiplatform, storage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
for _line in (_ROOT / ".env").read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.environ.get("GCP_REGION", "us-central1")
BUCKET = os.environ["GCS_BUCKET_NAME"]
ENDPOINT_ID = "1041396390934085632"

ARTIFACT_URI = f"gs://{BUCKET}/models/v1"
CONTAINER_IMAGE_URI = (
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/predictive-maintenance/rul-predictor:v1"
)

LOCAL_SCALER_PARAMS = _ROOT / "model_artifacts" / "scaler_params.npz"
GCS_SCALER_BLOB = "models/v1/scaler_params.npz"


# ---------------------------------------------------------------------------
# Step 1: Upload scaler_params.npz if missing from GCS
# ---------------------------------------------------------------------------
def ensure_scaler_params() -> None:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blob = bucket.blob(GCS_SCALER_BLOB)

    if blob.exists():
        print(f"✓ scaler_params.npz already in GCS ({blob.size} bytes)")
        return

    if not LOCAL_SCALER_PARAMS.exists():
        print(
            f"ERROR: {LOCAL_SCALER_PARAMS} not found locally.\n"
            "Run scripts/extract_scaler_params.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Uploading {LOCAL_SCALER_PARAMS} → gs://{BUCKET}/{GCS_SCALER_BLOB} ...")
    blob.upload_from_filename(str(LOCAL_SCALER_PARAMS))
    print("Upload complete.")


# ---------------------------------------------------------------------------
# Step 2: Register model with custom container
# ---------------------------------------------------------------------------
def register_model() -> aiplatform.Model:
    print("\nRegistering model with custom container...")
    print(f"  artifact_uri  : {ARTIFACT_URI}")
    print(f"  container_uri : {CONTAINER_IMAGE_URI}")

    model = aiplatform.Model.upload(
        display_name="rul-predictor-custom-v1",
        artifact_uri=ARTIFACT_URI,
        serving_container_image_uri=CONTAINER_IMAGE_URI,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        serving_container_environment_variables={},
        description="RUL LSTM — custom FastAPI container (no TorchServe)",
    )
    print(f"Model registered: {model.resource_name}")
    return model


# ---------------------------------------------------------------------------
# Step 3: Deploy to existing endpoint
# ---------------------------------------------------------------------------
def deploy_model(model: aiplatform.Model) -> None:
    endpoint = aiplatform.Endpoint(
        f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"
    )
    print(f"\nDeploying to endpoint {ENDPOINT_ID}...")
    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=2,
        traffic_percentage=100,
        sync=False,
    )
    print("Deployment submitted (async).")
    print(
        "Monitor: gcloud ai operations list --region=us-central1 --project="
        + PROJECT_ID
    )
    print("\nAfter ~15 min check logs:")
    print(
        f"  gcloud logging read 'resource.labels.endpoint_id=\"{ENDPOINT_ID}\"' "
        f"--project={PROJECT_ID} --freshness=30m --order=asc 2>/dev/null | tail -50"
    )


def main() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)
    ensure_scaler_params()
    model = register_model()
    deploy_model(model)
    print("\n✅ Done. Deployment in progress — check logs in ~15 minutes.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
