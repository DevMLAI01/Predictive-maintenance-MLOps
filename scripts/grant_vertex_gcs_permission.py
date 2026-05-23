"""Grant Vertex AI service accounts access to the GCS bucket.

Grants two SAs:
  1. Vertex AI SA  (model registry / endpoint)   → objectViewer
  2. Compute Engine default SA (Pipelines runner) → objectAdmin (read + write artifacts)

Run once before register_model.py and run_pipeline.py:

    python scripts/grant_vertex_gcs_permission.py
"""

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

GCP_PROJECT_NUMBER = "350955102644"

GRANTS = [
    # Vertex AI model-serving SA — needs to read model artifacts
    (
        f"serviceAccount:service-{GCP_PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com",
        "roles/storage.objectViewer",
    ),
    # Compute Engine default SA — Vertex AI Pipelines runner needs read+write
    (
        f"serviceAccount:{GCP_PROJECT_NUMBER}-compute@developer.gserviceaccount.com",
        "roles/storage.objectAdmin",
    ),
]


def grant() -> None:
    client = storage.Client(project=GCP_PROJECT_ID)
    bucket = client.bucket(GCS_BUCKET_NAME)
    policy = bucket.get_iam_policy(requested_policy_version=3)

    changed = False
    for member, role in GRANTS:
        already = any(
            b["role"] == role and member in b["members"] for b in policy.bindings
        )
        if already:
            print(f"  already set : {member} → {role}")
        else:
            policy.bindings.append({"role": role, "members": {member}})
            print(f"  granting    : {member} → {role}")
            changed = True

    if changed:
        bucket.set_iam_policy(policy)
        print(f"IAM policy updated on gs://{GCS_BUCKET_NAME}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    grant()
