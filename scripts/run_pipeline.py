"""Compile and submit the RUL KFP v2 pipeline to Vertex AI Pipelines.

Run from the repo root:

    python scripts/run_pipeline.py

The script will:
  1. Compile the pipeline to rul_pipeline.yaml
  2. Upload the YAML to GCS
  3. Submit a PipelineJob to Vertex AI
  4. Print the GCP Console URL for live monitoring
  5. Poll every 60 s until completion, then print the final status
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from google.cloud import aiplatform, storage
from kfp import compiler

# Add repo root to path so "src.pipeline.pipeline" is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pipeline.pipeline import rul_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_ENV_FILE = _ROOT / ".env"
for _line in _ENV_FILE.read_text().splitlines():
    if _line.strip() and not _line.startswith("#") and "=" in _line:
        _k, _, _v = _line.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

GCP_PROJECT_ID = os.environ["GCP_PROJECT_ID"]
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]
VERTEX_MODEL_ID = os.environ.get("VERTEX_MODEL_ID", "")

PIPELINE_YAML = _ROOT / "rul_pipeline.yaml"
GCS_PIPELINE_YAML = f"gs://{GCS_BUCKET_NAME}/pipelines/rul_pipeline.yaml"
VERTEX_MODEL_RESOURCE = (
    f"projects/{GCP_PROJECT_ID}/locations/{GCP_REGION}/models/{VERTEX_MODEL_ID}"
)


# ---------------------------------------------------------------------------
# Step 1: Compile
# ---------------------------------------------------------------------------
def compile_pipeline() -> None:
    print("Compiling pipeline → rul_pipeline.yaml ...")
    compiler.Compiler().compile(
        pipeline_func=rul_pipeline,
        package_path=str(PIPELINE_YAML),
    )
    size_kb = PIPELINE_YAML.stat().st_size / 1024
    print(f"  Compiled ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Step 2: Upload YAML to GCS
# ---------------------------------------------------------------------------
def upload_yaml() -> None:
    print(f"Uploading → {GCS_PIPELINE_YAML} ...")
    uri = GCS_PIPELINE_YAML[5:]
    bucket_name, _, blob_path = uri.partition("/")
    storage.Client(project=GCP_PROJECT_ID).bucket(bucket_name).blob(
        blob_path
    ).upload_from_filename(str(PIPELINE_YAML))
    print("  Uploaded.")


# ---------------------------------------------------------------------------
# Step 3: Submit pipeline job
# ---------------------------------------------------------------------------
def submit_pipeline() -> aiplatform.PipelineJob:
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

    job = aiplatform.PipelineJob(
        display_name="rul-predictor-pipeline",
        template_path=str(PIPELINE_YAML),
        pipeline_root=f"gs://{GCS_BUCKET_NAME}/pipelines/",
        parameter_values={
            "gcs_raw_data_uri": f"gs://{GCS_BUCKET_NAME}/data/raw",
            "gcs_model_uri": f"gs://{GCS_BUCKET_NAME}/models/v1",
            "vertex_model_resource_name": VERTEX_MODEL_RESOURCE,
            "window_size": 30,
            "rul_cap": 125,
            "epochs": 100,
            "batch_size": 256,
            "rmse_threshold": 18.0,
            "machine_type": "n1-standard-2",
            "endpoint_display_name": "rul-predictor-endpoint",
        },
        labels={
            "project": "predictive-maintenance",
            "dataset": "nasa-cmapss-fd001",
        },
        enable_caching=True,
    )

    job.submit()
    return job


# ---------------------------------------------------------------------------
# Step 4: Monitor until completion
# ---------------------------------------------------------------------------
def monitor(job: aiplatform.PipelineJob) -> None:
    console_url = (
        f"https://console.cloud.google.com/vertex-ai/pipelines/runs/"
        f"{job.resource_name.split('/')[-1]}"
        f"?project={GCP_PROJECT_ID}"
    )
    print()
    print("=" * 70)
    print(f"Pipeline submitted: {job.resource_name}")
    print(f"Monitor in Console: {console_url}")
    print("=" * 70)
    print("Polling every 60 s  (Ctrl-C to stop polling — job continues in GCP)")
    print()

    terminal_states = {
        "PIPELINE_STATE_SUCCEEDED",
        "PIPELINE_STATE_FAILED",
        "PIPELINE_STATE_CANCELLED",
    }
    dots = 0
    while True:
        try:
            time.sleep(60)
            # Re-fetch from Vertex AI to get the latest state
            current = aiplatform.PipelineJob.get(job.resource_name)
            state = current.state.name
            dots += 1
            print(f"  [{dots:>3}m] {state}")
            if state in terminal_states:
                job = current  # use latest for final reporting
                break
        except KeyboardInterrupt:
            print("\nStopped polling. Pipeline still running in Vertex AI.")
            return

    print()
    if job.state.name == "PIPELINE_STATE_SUCCEEDED":
        print("✅ Pipeline SUCCEEDED")
        # Try to surface the endpoint URI from the deploy step
        for task in job.task_details:
            if task.task_name == "deploy-model":
                outputs = task.outputs
                if "endpoint_uri" in outputs:
                    uri = outputs["endpoint_uri"].artifacts[0].uri
                    print(f"   Endpoint URI: {uri}")
                    print(f"   Add to .env:  STREAMLIT_ENDPOINT_URL={uri}")
    else:
        print(f"❌ Pipeline ended with state: {job.state.name}")
        print(f"   Check logs at: {console_url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not VERTEX_MODEL_ID:
        print(
            "ERROR: VERTEX_MODEL_ID not set in .env — run Phase 4 first.",
            file=sys.stderr,
        )
        sys.exit(1)

    compile_pipeline()
    upload_yaml()
    job = submit_pipeline()
    monitor(job)


if __name__ == "__main__":
    main()
