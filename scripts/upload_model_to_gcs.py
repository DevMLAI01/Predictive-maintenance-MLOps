"""Upload trained model artifacts to GCS. Run from Cloud Shell (pre-authenticated)."""

from pathlib import Path

from google.cloud import storage

BUCKET_NAME = "predictive-maintenance-artifacts"
GCS_PREFIX = "models/v1"
PROJECT_ID = "groovy-rope-496901-d2"

ARTIFACTS = [
    "model_artifacts/rul_predictor_v1.pt",
    "model_artifacts/scaler.joblib",
    "model_artifacts/evaluation_report.json",
]


def upload_artifacts(local_dir: str = ".") -> None:
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    for rel_path in ARTIFACTS:
        local_path = Path(local_dir) / rel_path
        if not local_path.exists():
            print(f"  SKIP (not found): {local_path}")
            continue

        blob_name = f"{GCS_PREFIX}/{local_path.name}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))
        size_mb = local_path.stat().st_size / 1e6
        print(
            f"  OK  {local_path.name} ({size_mb:.2f} MB) -> gs://{BUCKET_NAME}/{blob_name}"
        )

    print("\nUpload complete.")
    print(f"Artifacts at: gs://{BUCKET_NAME}/{GCS_PREFIX}/")


if __name__ == "__main__":
    upload_artifacts()
