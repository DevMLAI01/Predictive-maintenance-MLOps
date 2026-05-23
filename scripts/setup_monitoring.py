"""Configure Vertex AI Model Monitoring for the RUL prediction endpoint.

Monitors:
  - Feature skew  : sensor value distribution shift (15 retained sensors)
  - Prediction drift : RUL output distribution shift

Skew threshold of 0.3 uses Jensen-Shannon divergence.  Alerts are written to
Cloud Logging and visible in Console > Vertex AI > Model Monitoring.

Usage (run in Cloud Shell after Phase 5/6 endpoint is deployed):
    python scripts/setup_monitoring.py

Portfolio note:
  This mirrors the LLM Drift Monitor project narrative — "I build systems that
  detect when things go wrong, whether ML models drifting or physical assets
  degrading."  Both use the same underlying principle: establish a baseline
  distribution and alert when production diverges.

Verification:
  Console > Vertex AI > Model Monitoring > rul-predictor-monitoring
  Check that state = RUNNING and objectives show 15 sensor thresholds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import JobServiceClient
from google.cloud.aiplatform_v1.types import (
    ModelDeploymentMonitoringJob,
    ModelDeploymentMonitoringObjectiveConfig,
    ModelDeploymentMonitoringScheduleConfig,
    ModelMonitoringAlertConfig,
    ModelMonitoringObjectiveConfig,
    SamplingStrategy,
    ThresholdConfig,
)
from google.protobuf.duration_pb2 import Duration
from google.protobuf.field_mask_pb2 import FieldMask

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
DEPLOYED_MODEL_ID = os.environ.get("DEPLOYED_MODEL_ID", "3482694834643271680")

ENDPOINT_RESOURCE = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}"

# ---------------------------------------------------------------------------
# Monitoring thresholds
# ---------------------------------------------------------------------------

# 15 retained sensors after dropping zero-variance / NaN-correlation ones.
# FD001 sensors kept: 2,3,4,6,7,8,9,11,12,13,14,15,17,20,21
_RETAINED_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

# Jensen-Shannon divergence threshold — 0.3 flags moderate distribution shift.
# Lower = more sensitive; 0.5 = near-complete distribution change.
_SKEW_THRESHOLD = 0.3
_DRIFT_THRESHOLD = 0.3

# Alert probability threshold for failure_probability feature
_FAILURE_PROB_DRIFT_THRESHOLD = 0.2

# Sampling rate for prediction logging (10% → reduces BigQuery cost)
_SAMPLE_RATE = 0.1

# Monitoring cadence
_MONITOR_INTERVAL_SECONDS = 3600  # hourly


# ---------------------------------------------------------------------------
# Build monitoring configuration
# ---------------------------------------------------------------------------


def _feature_skew_thresholds() -> dict[str, ThresholdConfig]:
    """Return a JS-divergence threshold for each retained sensor."""
    return {name: ThresholdConfig(value=_SKEW_THRESHOLD) for name in _RETAINED_SENSORS}


def _prediction_drift_thresholds() -> dict[str, ThresholdConfig]:
    """Return drift thresholds for model output features."""
    return {
        "rul": ThresholdConfig(value=_DRIFT_THRESHOLD),
        "failure_prob": ThresholdConfig(value=_FAILURE_PROB_DRIFT_THRESHOLD),
    }


def _build_objective_config() -> ModelDeploymentMonitoringObjectiveConfig:
    """Build per-model objective config covering sensor drift + prediction drift.

    Uses prediction_drift_detection_config for all features (sensors and outputs)
    so no training dataset URI is required.
    """
    all_thresholds = {**_feature_skew_thresholds(), **_prediction_drift_thresholds()}
    objective = ModelMonitoringObjectiveConfig(
        prediction_drift_detection_config=(
            ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
                drift_thresholds=all_thresholds,
            )
        ),
    )

    return ModelDeploymentMonitoringObjectiveConfig(
        deployed_model_id=DEPLOYED_MODEL_ID,
        objective_config=objective,
    )


def setup_monitoring() -> ModelDeploymentMonitoringJob:
    """Create and submit the model monitoring job."""

    client = JobServiceClient(
        client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
    )

    job = ModelDeploymentMonitoringJob(
        display_name="rul-predictor-monitoring",
        endpoint=ENDPOINT_RESOURCE,
        model_deployment_monitoring_objective_configs=[_build_objective_config()],
        model_deployment_monitoring_schedule_config=(
            ModelDeploymentMonitoringScheduleConfig(
                monitor_interval=Duration(seconds=_MONITOR_INTERVAL_SECONDS),
            )
        ),
        logging_sampling_strategy=SamplingStrategy(
            random_sample_config=SamplingStrategy.RandomSampleConfig(
                sample_rate=_SAMPLE_RATE,
            )
        ),
        model_monitoring_alert_config=ModelMonitoringAlertConfig(
            enable_logging=True,  # write anomalies to Cloud Logging
        ),
        enable_monitoring_pipeline_logs=True,
    )

    parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    print("Checking for existing monitoring job...")
    print(f"  Endpoint  : {ENDPOINT_RESOURCE}")
    print(f"  Model ID  : {DEPLOYED_MODEL_ID}")
    print(f"  Interval  : every {_MONITOR_INTERVAL_SECONDS // 60} minutes")
    print(f"  Sampling  : {int(_SAMPLE_RATE * 100)}% of predictions logged")
    print(f"  Drift threshold (sensors)        : {_SKEW_THRESHOLD}")
    print(f"  Drift threshold (failure_prob)   : {_FAILURE_PROB_DRIFT_THRESHOLD}")

    existing = next(
        (
            j
            for j in client.list_model_deployment_monitoring_jobs(parent=parent)
            if j.endpoint == ENDPOINT_RESOURCE
        ),
        None,
    )

    if existing is not None:
        print(f"  → Found existing job {existing.name.split('/')[-1]} — updating...")
        job.name = existing.name
        operation = client.update_model_deployment_monitoring_job(
            model_deployment_monitoring_job=job,
            update_mask=FieldMask(
                paths=["model_deployment_monitoring_objective_configs"]
            ),
        )
        response = operation.result()
        print("  → Updated.")
    else:
        print("  → No existing job found — creating...")
        response = client.create_model_deployment_monitoring_job(
            parent=parent,
            model_deployment_monitoring_job=job,
        )
        print("  → Created.")

    return response


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def print_monitoring_summary(job: ModelDeploymentMonitoringJob) -> None:
    """Print resource name and console link for the created monitoring job."""
    job_name = job.name  # projects/.../locations/.../modelDeploymentMonitoringJobs/NNN
    job_id = job_name.split("/")[-1]

    console_url = (
        f"https://console.cloud.google.com/vertex-ai/model-monitoring"
        f"?project={PROJECT_ID}"
    )

    print(f"\n{'=' * 60}")
    print("✅ Monitoring job created")
    print(f"   Resource : {job_name}")
    print(f"   Job ID   : {job_id}")
    print(f"   State    : {job.state.name}")
    print("\nVerify in GCP Console:")
    print(f"   {console_url}")
    print("\nView anomaly alerts in Cloud Logging:")
    print(
        f"   gcloud logging read 'logName:\"aiplatform.googleapis.com\"' "
        f"--project={PROJECT_ID} --freshness=24h | grep -i drift"
    )
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    aiplatform.init(project=PROJECT_ID, location=REGION)

    print("\nVertex AI Model Monitoring Setup")
    print(f"  Project  : {PROJECT_ID}")
    print(f"  Region   : {REGION}")
    print(f"  Endpoint : {ENDPOINT_ID}\n")

    monitoring_job = setup_monitoring()
    print_monitoring_summary(monitoring_job)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print(
            "\nCommon causes:",
            "\n  - Endpoint not deployed (run Phase 5 pipeline first)",
            "\n  - DEPLOYED_MODEL_ID stale (check .env after redeployment)",
            "\n  - Insufficient IAM permissions (need roles/aiplatform.admin)",
            file=sys.stderr,
        )
        sys.exit(1)
