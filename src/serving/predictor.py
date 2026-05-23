"""Vertex AI custom predictor for the RUL LSTM model.

Implements the Predictor interface consumed by the Vertex AI pre-built
PyTorch serving container (pytorch-cpu.2-0 / pytorch-gpu.2-0).
"""

from __future__ import annotations

import logging
import os
import sys

import joblib
import numpy as np
import torch

# Allow importing the model definition when running inside the container.
# The model source is expected alongside this file or on PYTHONPATH.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lstm import RULPredictor  # noqa: E402

from google.cloud.aiplatform.prediction.predictor import Predictor  # noqa: E402
from google.cloud import storage  # noqa: E402

logger = logging.getLogger(__name__)

_N_FEATURES = 15
_WINDOW_SIZE = 30
_RUL_CAP = 125.0


class RULCustomPredictor(Predictor):
    """Vertex AI Predictor for dual-head LSTM RUL + failure-probability model."""

    def __init__(self) -> None:
        self._model: RULPredictor | None = None
        self._scaler = None

    # ------------------------------------------------------------------
    # Vertex AI lifecycle hooks
    # ------------------------------------------------------------------

    def load(self, artifacts_uri: str) -> None:
        """Download artifacts from GCS and initialise model + scaler.

        Parameters
        ----------
        artifacts_uri:
            GCS URI of the model directory, e.g.
            ``gs://predictive-maintenance-artifacts/models/v1/``
        """
        logger.info("Loading model from %s", artifacts_uri)

        local_dir = "/tmp/model_artifacts"
        os.makedirs(local_dir, exist_ok=True)

        self._download_from_gcs(artifacts_uri, local_dir)

        model_path = os.path.join(local_dir, "rul_predictor_v1.pt")
        scaler_path = os.path.join(local_dir, "scaler.joblib")

        self._model = RULPredictor(
            n_features=_N_FEATURES,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            window_size=_WINDOW_SIZE,
        )
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        self._scaler = joblib.load(scaler_path)

        logger.info("Model loaded from %s", artifacts_uri)

    # ------------------------------------------------------------------
    # Prediction pipeline
    # ------------------------------------------------------------------

    def preprocess(self, prediction_input: dict) -> torch.Tensor:
        """Convert raw JSON payload to a scaled model-ready tensor.

        Expected input format::

            {
              "instances": [
                [[s1, s2, ..., s15], ...],   # 30 x 15 sensor readings per engine
                ...
              ]
            }

        Returns
        -------
        torch.Tensor of shape ``(batch, 30, 15)``
        """
        instances = prediction_input.get("instances", [])
        arr = np.array(instances, dtype=np.float32)  # (batch, 30, 15)

        batch, seq_len, n_feat = arr.shape
        flat = arr.reshape(-1, n_feat)  # (batch*30, 15)
        flat_scaled = self._scaler.transform(flat)
        scaled = flat_scaled.reshape(batch, seq_len, n_feat)

        return torch.tensor(scaled, dtype=torch.float32)

    def predict(self, instances: torch.Tensor) -> dict:
        """Run the LSTM forward pass.

        Parameters
        ----------
        instances:
            Tensor ``(batch, 30, 15)`` from :meth:`preprocess`.

        Returns
        -------
        dict with keys ``rul_predicted`` and ``failure_probability``
        (lists, one entry per batch item).
        """
        with torch.no_grad():
            outputs = self._model(instances)

        rul_raw = outputs["rul"].squeeze(-1).tolist()  # [batch]
        fail_prob = outputs["failure_prob"].squeeze(-1).tolist()  # [batch]

        # Ensure scalars are wrapped in lists when batch=1
        if isinstance(rul_raw, float):
            rul_raw = [rul_raw]
            fail_prob = [fail_prob]

        return {
            "rul_predicted": rul_raw,
            "failure_probability": fail_prob,
        }

    def postprocess(self, prediction_results: dict) -> dict:
        """Format model outputs for the JSON response.

        Returns
        -------
        ::

            {
              "predictions": [
                {
                  "rul": int,
                  "failure_prob": float,
                  "alert": bool,
                  "confidence": "high" | "medium" | "critical"
                },
                ...
              ]
            }
        """
        predictions = []
        for rul_raw, fp in zip(
            prediction_results["rul_predicted"],
            prediction_results["failure_probability"],
        ):
            rul_clamped = float(np.clip(rul_raw, 0.0, _RUL_CAP))
            rul_int = int(round(rul_clamped))
            fp_float = round(float(fp), 4)
            alert = fp_float > 0.7

            if rul_int > 50:
                confidence = "high"
            elif rul_int >= 20:
                confidence = "medium"
            else:
                confidence = "critical"

            predictions.append(
                {
                    "rul": rul_int,
                    "failure_prob": fp_float,
                    "alert": alert,
                    "confidence": confidence,
                }
            )

        return {"predictions": predictions}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_from_gcs(gcs_uri: str, local_dir: str) -> None:
        """Download all blobs under *gcs_uri* into *local_dir*."""
        # Strip trailing slash and parse bucket / prefix
        gcs_uri = gcs_uri.rstrip("/")
        without_scheme = gcs_uri[len("gs://") :]
        bucket_name, _, prefix = without_scheme.partition("/")

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for blob in client.list_blobs(bucket_name, prefix=prefix):
            filename = os.path.basename(blob.name)
            if not filename:
                continue
            dest = os.path.join(local_dir, filename)
            blob.download_to_filename(dest)
            logger.info("Downloaded %s → %s", blob.name, dest)
