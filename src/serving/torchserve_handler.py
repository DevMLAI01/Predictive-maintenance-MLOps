"""TorchServe custom handler for the RUL LSTM model.

Dependencies: only torch + numpy (both pre-installed in pytorch-cpu container).
Scaler parameters are stored as a .npz file — no joblib or scikit-learn needed.
"""

import sys

print(f"[RUL-IMPORT] handler module loading, Python {sys.version}", flush=True)

import json
import logging
import os
import traceback

print("[RUL-IMPORT] stdlib imports done", flush=True)

import numpy as np

print(f"[RUL-IMPORT] numpy {np.__version__} imported OK", flush=True)

import torch

print(f"[RUL-IMPORT] torch {torch.__version__} imported OK", flush=True)

from ts.torch_handler.base_handler import BaseHandler

print("[RUL-IMPORT] BaseHandler imported OK", flush=True)

logger = logging.getLogger(__name__)

_N_FEATURES = 15
_WINDOW_SIZE = 30
_RUL_CAP = 125.0


class RULHandler(BaseHandler):
    def initialize(self, context) -> None:
        try:
            model_dir = context.system_properties.get("model_dir")
            print(f"[RUL] initialize called, model_dir={model_dir}", flush=True)
            print(f"[RUL] files: {os.listdir(model_dir)}", flush=True)

            sys.path.insert(0, model_dir)
            print("[RUL] importing RULPredictor", flush=True)
            from lstm import RULPredictor

            print("[RUL] building model", flush=True)
            self.model = RULPredictor(
                n_features=_N_FEATURES,
                hidden_size=128,
                num_layers=2,
                dropout=0.2,
                window_size=_WINDOW_SIZE,
            )
            pt_path = os.path.join(model_dir, "rul_predictor_v1.pt")
            print(f"[RUL] loading weights from {pt_path}", flush=True)
            state = torch.load(pt_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()

            npz_path = os.path.join(model_dir, "scaler_params.npz")
            print(f"[RUL] loading scaler from {npz_path}", flush=True)
            params = np.load(npz_path)
            self._scale = params["scale_"].astype(np.float32)
            self._min = params["min_"].astype(np.float32)

            self.initialized = True
            print("[RUL] initialize COMPLETE", flush=True)
        except Exception as exc:
            print(f"[RUL] initialize FAILED: {exc}", flush=True)
            print(traceback.format_exc(), flush=True)
            raise

    def _scale_input(self, arr: np.ndarray) -> np.ndarray:
        batch, seq, feat = arr.shape
        flat = arr.reshape(-1, feat)
        flat_scaled = flat * self._scale + self._min
        return flat_scaled.reshape(batch, seq, feat)

    def preprocess(self, data) -> torch.Tensor:
        body = data[0].get("body", {})
        if isinstance(body, (bytes, bytearray)):
            body = json.loads(body)
        instances = body.get("instances", [])
        arr = np.array(instances, dtype=np.float32)
        scaled = self._scale_input(arr)
        return torch.tensor(scaled, dtype=torch.float32)

    def inference(self, inputs: torch.Tensor) -> dict:
        with torch.no_grad():
            return self.model(inputs)

    def postprocess(self, outputs: dict) -> list:
        rul_list = outputs["rul"].squeeze(-1).tolist()
        fp_list = outputs["failure_prob"].squeeze(-1).tolist()

        if isinstance(rul_list, float):
            rul_list, fp_list = [rul_list], [fp_list]

        predictions = []
        for rul_raw, fp in zip(rul_list, fp_list):
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

        return [json.dumps({"predictions": predictions})]
