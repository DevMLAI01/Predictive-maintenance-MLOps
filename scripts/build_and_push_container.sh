#!/bin/bash
# Build and push the RUL predictor custom container via Cloud Build.
#
# Run this in Google Cloud Shell from the project root:
#   bash scripts/build_and_push_container.sh
#
# Prerequisites (already done):
#   gcloud auth login / application-default login
#   Billing enabled on project

set -euo pipefail

PROJECT_ID="groovy-rope-496901-d2"
REGION="us-central1"
REPO="predictive-maintenance"
IMAGE="rul-predictor"
TAG="v1"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}"

echo "=== Step 1: Ensure Artifact Registry repository exists ==="
gcloud artifacts repositories describe "${REPO}" \
    --location="${REGION}" --project="${PROJECT_ID}" 2>/dev/null \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker \
       --location="${REGION}" \
       --project="${PROJECT_ID}"
echo "Repository: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}"

echo ""
echo "=== Step 2: Build and push via Cloud Build (no local Docker needed) ==="
echo "Image: ${IMAGE_URI}"
echo "This typically takes 5-10 minutes (PyTorch download is ~700 MB)."
echo ""

# Upload source (project root) + build using the Dockerfile in src/serving/
gcloud builds submit \
    --project="${PROJECT_ID}" \
    --tag="${IMAGE_URI}" \
    --dockerfile="src/serving/Dockerfile" \
    --timeout="20m" \
    .

echo ""
echo "✅ Container pushed: ${IMAGE_URI}"
echo ""
echo "Next step (run locally):"
echo "  python scripts/register_and_deploy_custom.py"
