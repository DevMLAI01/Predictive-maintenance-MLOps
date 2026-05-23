"""Validate the deployed Vertex AI RUL prediction endpoint.

Tests performed:
  1. Single prediction — latency < 200ms, response schema valid
  2. Batch predictions — 10 sequential requests, all succeed
  3. Response schema — required keys, correct types and ranges

Usage (run in Cloud Shell or locally with ADC):
    python scripts/test_endpoint.py

Expected output:
    Endpoint healthy: 3/3 tests passed
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from google.cloud import aiplatform

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
for _line in (_ROOT / ".env").read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
REGION = os.environ.get("GCP_REGION", "us-central1")
ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID", "1041396390934085632")

_N_FEATURES = 15
_WINDOW_SIZE = 30
_LATENCY_TARGET_MS = 200


# ---------------------------------------------------------------------------
# EndpointTester
# ---------------------------------------------------------------------------


class EndpointTester:
    """Send prediction requests to the Vertex AI RUL endpoint and validate responses."""

    def __init__(self, endpoint_id: str, project: str, region: str) -> None:
        aiplatform.init(project=project, location=region)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
        self._instances = self._load_test_instances()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_test_instances(self) -> list[list[list[float]]]:
        """Load raw (unscaled) sensor windows from local processed data.

        X_test.npy contains MinMaxScaler-normalized values.
        We inverse-transform to raw sensor readings before sending to the
        endpoint, which applies the scaler internally.
        """
        x_test_path = _ROOT / "data" / "processed" / "X_test.npy"
        scaler_path = _ROOT / "model_artifacts" / "scaler.joblib"

        if x_test_path.exists() and scaler_path.exists():
            import joblib

            X_scaled = np.load(str(x_test_path))  # (100, 30, 15)
            scaler = joblib.load(str(scaler_path))

            batch, seq, feat = X_scaled.shape
            flat_raw = scaler.inverse_transform(X_scaled.reshape(-1, feat))
            X_raw = flat_raw.reshape(batch, seq, feat)

            print(f"  Loaded {batch} test instances from local processed data.")
            return X_raw.tolist()

        # Fallback: mid-range synthetic sensor values (0.5 after scaling ≈ centre)
        print("  [WARNING] Local processed data not found — using synthetic instances.")
        return [[[0.5] * _N_FEATURES for _ in range(_WINDOW_SIZE)] for _ in range(10)]

    # ------------------------------------------------------------------
    # Individual tests
    # ------------------------------------------------------------------

    def test_single_prediction(self) -> dict[str, Any]:
        """Call the endpoint with one engine window and record latency.

        Sends a warm-up request first so cold-start time is excluded from
        the latency measurement (cold start can be 500ms+ on first call).
        """
        self.endpoint.predict(instances=[self._instances[0]])  # warm-up
        start = time.perf_counter()
        response = self.endpoint.predict(instances=[self._instances[0]])
        latency_ms = (time.perf_counter() - start) * 1000

        pred = response.predictions[0]
        print(f"\n  RUL              : {pred['rul']} cycles")
        print(f"  Failure prob     : {pred['failure_prob']:.4f}")
        print(f"  Maintenance alert: {pred['alert']}")
        print(f"  Confidence       : {pred['confidence']}")
        print(f"  Latency          : {latency_ms:.0f} ms")

        return {"prediction": pred, "latency_ms": latency_ms}

    def test_batch_predictions(self, n_samples: int = 10) -> pd.DataFrame:
        """Send n_samples sequential prediction requests and return a results DataFrame."""
        records: list[dict] = []
        instances = self._instances[:n_samples]

        print(
            f"\n  {'#':>3}  {'RUL':>6}  {'Prob':>6}  {'Alert':>5}  {'Confidence':<9}  {'Latency':>8}"
        )
        print(f"  {'-' * 3}  {'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 9}  {'-' * 8}")

        for i, instance in enumerate(instances):
            start = time.perf_counter()
            response = self.endpoint.predict(instances=[instance])
            latency_ms = (time.perf_counter() - start) * 1000
            pred = response.predictions[0]

            records.append(
                {
                    "sample": i + 1,
                    "rul": pred["rul"],
                    "failure_prob": pred["failure_prob"],
                    "alert": pred["alert"],
                    "confidence": pred["confidence"],
                    "latency_ms": round(latency_ms, 1),
                }
            )
            alert_str = "YES" if pred["alert"] else "no"
            print(
                f"  {i + 1:>3}  {pred['rul']:>6}  {pred['failure_prob']:>6.4f}"
                f"  {alert_str:>5}  {pred['confidence']:<9}  {latency_ms:>7.0f}ms"
            )

        df = pd.DataFrame(records)
        p50 = df["latency_ms"].median()
        p95 = df["latency_ms"].quantile(0.95)
        print(
            f"\n  Latency — p50: {p50:.0f}ms  p95: {p95:.0f}ms  (target: <{_LATENCY_TARGET_MS}ms)"
        )
        return df

    def validate_response_schema(self, prediction: dict[str, Any]) -> bool:
        """Return True if prediction dict matches the expected schema."""
        required_keys = {"rul", "failure_prob", "alert", "confidence"}
        valid_confidences = {"high", "medium", "critical"}
        errors: list[str] = []

        missing = required_keys - prediction.keys()
        if missing:
            errors.append(f"missing keys: {missing}")

        if "rul" in prediction and not (
            isinstance(prediction["rul"], (int, float)) and prediction["rul"] >= 0
        ):
            errors.append(
                f"rul must be a non-negative number, got {prediction['rul']!r}"
            )

        if "failure_prob" in prediction and not (
            0.0 <= prediction["failure_prob"] <= 1.0
        ):
            errors.append(
                f"failure_prob must be in [0, 1], got {prediction['failure_prob']!r}"
            )

        if "alert" in prediction and not isinstance(prediction["alert"], bool):
            errors.append(
                f"alert must be bool, got {type(prediction['alert']).__name__}"
            )

        if (
            "confidence" in prediction
            and prediction["confidence"] not in valid_confidences
        ):
            errors.append(
                f"confidence must be one of {valid_confidences}, got {prediction['confidence']!r}"
            )

        if errors:
            for e in errors:
                print(f"  SCHEMA ERROR: {e}")
            return False
        return True

    # ------------------------------------------------------------------
    # Test runner
    # ------------------------------------------------------------------

    def run_all_tests(self) -> None:
        """Execute all endpoint validation tests and print a summary."""
        print(f"\n{'=' * 60}")
        print("Endpoint Validation")
        print(f"  Endpoint ID : {ENDPOINT_ID}")
        print(f"  Project     : {PROJECT_ID}")
        print(f"  Region      : {REGION}")
        print(f"{'=' * 60}")

        results: dict[str, bool] = {}

        # ------------------------------------------------------------------
        # Test 1: Single prediction + latency
        # ------------------------------------------------------------------
        print("\n[TEST 1/3] Single prediction + latency check")
        try:
            result = self.test_single_prediction()
            schema_ok = self.validate_response_schema(result["prediction"])
            latency_ok = result["latency_ms"] < _LATENCY_TARGET_MS
            passed = schema_ok and latency_ok
            if not latency_ok:
                print(
                    f"  [WARN] Latency {result['latency_ms']:.0f}ms exceeds {_LATENCY_TARGET_MS}ms target"
                )
            results["single_prediction"] = passed
            print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
        except Exception as exc:
            print(f"  ❌ FAIL — {exc}")
            results["single_prediction"] = False

        # ------------------------------------------------------------------
        # Test 2: Batch predictions (10 requests)
        # ------------------------------------------------------------------
        print("\n[TEST 2/3] Batch predictions (10 sequential requests)")
        try:
            df = self.test_batch_predictions(n_samples=10)
            all_succeeded = len(df) == 10
            results["batch_predictions"] = all_succeeded
            print(
                f"  {'✅ PASS' if all_succeeded else '❌ FAIL'} — {len(df)}/10 predictions succeeded"
            )
        except Exception as exc:
            print(f"  ❌ FAIL — {exc}")
            results["batch_predictions"] = False

        # ------------------------------------------------------------------
        # Test 3: Response schema
        # ------------------------------------------------------------------
        print("\n[TEST 3/3] Response schema validation")
        try:
            response = self.endpoint.predict(instances=[self._instances[0]])
            pred = response.predictions[0]
            schema_ok = self.validate_response_schema(pred)
            results["schema_validation"] = schema_ok
            print(f"  {'✅ PASS' if schema_ok else '❌ FAIL'}")
        except Exception as exc:
            print(f"  ❌ FAIL — {exc}")
            results["schema_validation"] = False

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        n_passed = sum(results.values())
        n_total = len(results)
        print(f"\n{'=' * 60}")
        print(f"Endpoint healthy: {n_passed}/{n_total} tests passed")

        if n_passed == n_total:
            print("✅  Endpoint is healthy and ready for dashboard integration.")
        else:
            print("⚠️  Some tests failed — check endpoint container logs:")
            print(
                f"    gcloud logging read "
                f"'resource.labels.endpoint_id=\"{ENDPOINT_ID}\"' "
                f"--project={PROJECT_ID} --freshness=30m --order=asc | tail -50"
            )
        print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    tester = EndpointTester(
        endpoint_id=ENDPOINT_ID,
        project=PROJECT_ID,
        region=REGION,
    )
    tester.run_all_tests()


if __name__ == "__main__":
    main()
