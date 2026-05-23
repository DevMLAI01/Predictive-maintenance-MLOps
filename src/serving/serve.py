"""FastAPI prediction server for the RUL LSTM model.

Vertex AI custom container: listens on port 8080.
- GET  /health    → 200 OK
- POST /predict   → {"instances": [...]} → {"predictions": [...]}

On startup, downloads rul_predictor_v1.pt and scaler_params.npz from the
GCS path provided via AIP_STORAGE_URI (set automatically by Vertex AI).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, "/app")
from lstm import RULPredictor  # noqa: E402  (after sys.path patch)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_N_FEATURES = 15
_WINDOW_SIZE = 30
_RUL_CAP = 125.0
_MODEL_DIR = Path("/tmp/model")

_model: RULPredictor | None = None
_scale: np.ndarray | None = None
_min: np.ndarray | None = None


def _download_artifacts(storage_uri: str) -> None:
    from google.cloud import storage as gcs

    uri = storage_uri.rstrip("/").removeprefix("gs://")
    bucket_name, _, prefix = uri.partition("/")
    prefix = prefix.rstrip("/") + "/"

    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("rul_predictor_v1.pt", "scaler_params.npz"):
        blob_path = f"{prefix}{filename}"
        local = _MODEL_DIR / filename
        logger.info("Downloading gs://%s/%s → %s", bucket_name, blob_path, local)
        bucket.blob(blob_path).download_to_filename(str(local))
        logger.info("  done (%d bytes)", local.stat().st_size)


def _load_model() -> None:
    global _model, _scale, _min

    storage_uri = os.environ.get("AIP_STORAGE_URI", "").strip()
    if storage_uri:
        _download_artifacts(storage_uri)
    else:
        logger.warning("AIP_STORAGE_URI not set — expecting files in %s", _MODEL_DIR)

    pt_path = _MODEL_DIR / "rul_predictor_v1.pt"
    npz_path = _MODEL_DIR / "scaler_params.npz"

    if not pt_path.exists():
        raise FileNotFoundError(f"Model weights not found: {pt_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Scaler params not found: {npz_path}")

    logger.info("Loading model weights from %s", pt_path)
    _model = RULPredictor(
        n_features=_N_FEATURES,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        window_size=_WINDOW_SIZE,
    )
    state = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    _model.load_state_dict(state)
    _model.eval()

    params = np.load(str(npz_path))
    _scale = params["scale_"].astype(np.float32)
    _min = params["min_"].astype(np.float32)

    logger.info("Model loaded. Parameters: %d", _model.count_parameters())


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_model()
    logger.info("Server ready.")
    yield


app = FastAPI(lifespan=_lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(body: dict) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    instances = body.get("instances")
    if not instances:
        raise HTTPException(status_code=400, detail="'instances' key missing or empty")

    try:
        arr = np.array(instances, dtype=np.float32)  # (batch, 30, 15)
        if arr.ndim != 3 or arr.shape[1] != _WINDOW_SIZE or arr.shape[2] != _N_FEATURES:
            raise ValueError(
                f"Expected shape (batch, {_WINDOW_SIZE}, {_N_FEATURES}), got {arr.shape}"
            )

        # Inverse MinMaxScaler: x_original ≈ x_scaled * scale_ + min_
        batch, seq, feat = arr.shape
        flat = arr.reshape(-1, feat) * _scale + _min
        scaled = torch.tensor(flat.reshape(batch, seq, feat), dtype=torch.float32)

        with torch.no_grad():
            outputs = _model(scaled)

        rul_vals = outputs["rul"].squeeze(-1).tolist()
        fp_vals = outputs["failure_prob"].squeeze(-1).tolist()
        if isinstance(rul_vals, float):
            rul_vals, fp_vals = [rul_vals], [fp_vals]

        predictions = []
        for rul_raw, fp in zip(rul_vals, fp_vals):
            rul_int = int(round(float(np.clip(rul_raw, 0.0, _RUL_CAP))))
            fp_float = round(float(fp), 4)
            confidence = (
                "high" if rul_int > 50 else "medium" if rul_int >= 20 else "critical"
            )
            predictions.append(
                {
                    "rul": rul_int,
                    "failure_prob": fp_float,
                    "alert": fp_float > 0.7,
                    "confidence": confidence,
                }
            )

        return JSONResponse({"predictions": predictions})

    except (ValueError, RuntimeError) as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
