"""Create a TorchServe model archive (.mar) and upload it to GCS.

The pre-built Vertex AI PyTorch container (TorchServe) requires a model.mar
in the GCS artifact directory.  Run this once before register_model.py:

    pip install torch-model-archiver
    python scripts/create_and_upload_mar.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from google.cloud import storage

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _ROOT / ".env"
for _line in _ENV_FILE.read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
GCS_MAR_BLOB = "models/v1/model.mar"

# ---------------------------------------------------------------------------
# Local file paths
# ---------------------------------------------------------------------------
MODEL_PT = _ROOT / "model_artifacts" / "rul_predictor_v1.pt"
SCALER_PARAMS = _ROOT / "model_artifacts" / "scaler_params.npz"
HANDLER = _ROOT / "src" / "serving" / "torchserve_handler.py"
LSTM_DEF = _ROOT / "src" / "model" / "lstm.py"
TS_REQUIREMENTS = _ROOT / "src" / "serving" / "requirements.txt"


def check_archiver() -> None:
    if shutil.which("torch-model-archiver") is None:
        print("torch-model-archiver not found — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torch-model-archiver"]
        )


def build_mar(export_dir: Path) -> Path:
    cmd = [
        "torch-model-archiver",
        "--model-name",
        "model",  # → model.mar
        "--version",
        "1.0",
        "--serialized-file",
        str(MODEL_PT),
        "--handler",
        str(HANDLER),
        "--extra-files",
        f"{LSTM_DEF},{SCALER_PARAMS}",
        "--requirements-file",
        str(TS_REQUIREMENTS),
        "--export-path",
        str(export_dir),
        "--force",
    ]
    print("Building model archive...")
    print("  " + " ".join(cmd))
    subprocess.check_call(cmd)
    mar_path = export_dir / "model.mar"
    size_mb = mar_path.stat().st_size / 1e6
    print(f"Created {mar_path} ({size_mb:.1f} MB)")
    return mar_path


def upload_mar(mar_path: Path) -> None:
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob = bucket.blob(GCS_MAR_BLOB)
    print(f"Uploading → gs://{GCS_BUCKET_NAME}/{GCS_MAR_BLOB} ...")
    blob.upload_from_filename(str(mar_path))
    print("Upload complete.")


def main() -> None:
    check_archiver()
    with tempfile.TemporaryDirectory() as tmp:
        mar_path = build_mar(Path(tmp))
        upload_mar(mar_path)
    print()
    print("Next step: python scripts/register_model.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
