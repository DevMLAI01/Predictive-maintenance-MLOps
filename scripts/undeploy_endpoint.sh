#!/usr/bin/env bash
# Cost guardrail: undeploy the Vertex AI Endpoint after demos.
# Run in Cloud Shell when you're done demoing to stop per-hour billing.
# Re-deploy by re-running: python scripts/run_pipeline.py
#
# Usage:
#   bash scripts/undeploy_endpoint.sh

set -euo pipefail

source .env 2>/dev/null || true

GCP_PROJECT_ID="${GCP_PROJECT_ID:-groovy-rope-496901-d2}"
GCP_REGION="${GCP_REGION:-us-central1}"
VERTEX_ENDPOINT_ID="${VERTEX_ENDPOINT_ID:-}"

if [[ -z "$VERTEX_ENDPOINT_ID" ]]; then
    echo "ERROR: VERTEX_ENDPOINT_ID not set. Add it to .env or export it first."
    echo "  Find it at: https://console.cloud.google.com/vertex-ai/endpoints?project=$GCP_PROJECT_ID"
    exit 1
fi

echo "==> Undeploying all models from Vertex AI Endpoint: $VERTEX_ENDPOINT_ID"
echo "    Project: $GCP_PROJECT_ID | Region: $GCP_REGION"
echo ""

# List deployed models on the endpoint
DEPLOYED_MODEL_IDS=$(gcloud ai endpoints describe "$VERTEX_ENDPOINT_ID" \
    --project="$GCP_PROJECT_ID" \
    --region="$GCP_REGION" \
    --format="value(deployedModels[].id)" 2>/dev/null || echo "")

if [[ -z "$DEPLOYED_MODEL_IDS" ]]; then
    echo "No deployed models found on endpoint — nothing to undeploy."
    exit 0
fi

for DEPLOYED_MODEL_ID in $DEPLOYED_MODEL_IDS; do
    echo "  Undeploying model: $DEPLOYED_MODEL_ID ..."
    gcloud ai endpoints undeploy-model "$VERTEX_ENDPOINT_ID" \
        --project="$GCP_PROJECT_ID" \
        --region="$GCP_REGION" \
        --deployed-model-id="$DEPLOYED_MODEL_ID"
    echo "  ✓ $DEPLOYED_MODEL_ID undeployed."
done

echo ""
echo "==> Endpoint undeployed. Billing stopped."
echo ""
echo "To redeploy:"
echo "  python scripts/run_pipeline.py"
echo ""
echo "Note: Cloud Run dashboard scales to zero automatically (min-instances=0)."
echo "      No action needed for the Streamlit dashboard."
