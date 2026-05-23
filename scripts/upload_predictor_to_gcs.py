"""Upload src/serving/predictor.py to GCS (replaces gsutil cp for Phase 4.2).

Run from the repo root:

    python scripts/upload_predictor_to_gcs.py
"""

from __future__ import annotations

import os
from pathlib import Path

from google.cloud import storage

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
for _line in _ENV_FILE.read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

LOCAL_PATH = Path(__file__).resolve().parent.parent / "src" / "serving" / "predictor.py"
GCS_BLOB_PATH = "artifacts/predictor.py"


def upload() -> None:
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_BLOB_PATH)
    blob.upload_from_filename(str(LOCAL_PATH))
    print(f"Uploaded {LOCAL_PATH.name} → gs://{GCS_BUCKET_NAME}/{GCS_BLOB_PATH}")


if __name__ == "__main__":
    upload()
