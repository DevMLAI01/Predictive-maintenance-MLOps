"""Extract MinMaxScaler parameters to a numpy archive for TorchServe.

Saves scale_ and min_ arrays so the handler needs only numpy (no joblib/scikit-learn).

Run once before create_and_upload_mar.py:
    python scripts/extract_scaler_params.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCALER_IN = ROOT / "model_artifacts" / "scaler.joblib"
PARAMS_OUT = ROOT / "model_artifacts" / "scaler_params.npz"


def main() -> None:
    if not SCALER_IN.exists():
        print(f"ERROR: {SCALER_IN} not found", file=sys.stderr)
        sys.exit(1)

    scaler = joblib.load(SCALER_IN)
    np.savez(PARAMS_OUT, scale_=scaler.scale_, min_=scaler.min_)

    print(f"Saved {PARAMS_OUT}")
    print(f"  scale_ : {scaler.scale_.shape}  {scaler.scale_[:3].round(4)} ...")
    print(f"  min_   : {scaler.min_.shape}   {scaler.min_[:3].round(4)} ...")


if __name__ == "__main__":
    main()
