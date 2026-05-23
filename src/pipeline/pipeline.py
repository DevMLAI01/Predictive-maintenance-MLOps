"""KFP v2 pipeline definition for the RUL predictive-maintenance workflow.

Compiles to rul_pipeline.yaml via scripts/run_pipeline.py.
"""

import os
from pathlib import Path

from kfp import dsl

from src.pipeline.components import (
    deploy_model,
    engineer_features,
    evaluate_model,
    train_model,
    validate_data,
)

# Load env for the pipeline_root default
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"
for _line in _ENV_FILE.read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

_BUCKET = os.environ.get("GCS_BUCKET_NAME", "predictive-maintenance-artifacts")
_PROJECT = os.environ.get("GCP_PROJECT_ID", "")
_REGION = os.environ.get("GCP_REGION", "us-central1")


@dsl.pipeline(
    name="rul-predictor-pipeline",
    description="Predictive maintenance RUL pipeline — NASA CMAPSS FD001",
    pipeline_root=f"gs://{_BUCKET}/pipelines/",
)
def rul_pipeline(
    gcs_raw_data_uri: str = f"gs://{_BUCKET}/data/raw",
    gcs_model_uri: str = f"gs://{_BUCKET}/models/v1",
    vertex_model_resource_name: str = (
        f"projects/{_PROJECT}/locations/{_REGION}/models/"
        + os.environ.get("VERTEX_MODEL_ID", "")
    ),
    window_size: int = 30,
    rul_cap: int = 125,
    epochs: int = 100,
    batch_size: int = 256,
    rmse_threshold: float = 18.0,
    machine_type: str = "n1-standard-2",
    endpoint_display_name: str = "rul-predictor-endpoint",
) -> None:
    # ── Step 1: Validate data ──────────────────────────────────────────
    validation = validate_data(gcs_data_uri=gcs_raw_data_uri)
    validation.set_display_name("Validate Data")

    # ── Step 2+: Proceed only if data is valid ─────────────────────────
    with dsl.Condition(validation.outputs["is_valid"] == True, name="data-valid"):  # noqa: E712
        # ── Step 2: Feature engineering ───────────────────────────────
        features = engineer_features(
            gcs_data_uri=gcs_raw_data_uri,
            window_size=window_size,
            rul_cap=rul_cap,
        )
        features.set_display_name("Feature Engineering")

        # ── Step 3: Train model ───────────────────────────────────────
        training = train_model(
            processed_data=features.outputs["processed_data"],
            gcs_model_uri=gcs_model_uri,
            epochs=epochs,
            batch_size=batch_size,
        )
        training.set_display_name("Train LSTM Model")
        training.set_memory_limit("8G")

        # ── Step 4: Evaluate model ────────────────────────────────────
        evaluation = evaluate_model(
            model_artifact=training.outputs["model_artifact"],
            test_data_uri=gcs_raw_data_uri,
            rmse_threshold=rmse_threshold,
        )
        evaluation.set_display_name("Evaluate Model")

        # ── Step 5: Deploy only if RMSE passes threshold ──────────────
        with dsl.Condition(
            evaluation.outputs["passes_threshold"] == True,  # noqa: E712
            name="deploy-if-good",
        ):
            deployment = deploy_model(
                model_resource_name=vertex_model_resource_name,
                endpoint_display_name=endpoint_display_name,
                machine_type=machine_type,
            )
            deployment.set_display_name("Deploy to Vertex AI Endpoint")
