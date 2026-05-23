#!/bin/bash
# Phase 1.3 — GCP initialization
# Run this entirely in Google Cloud Shell (shell.cloud.google.com)
# Do NOT run locally — Cloud Shell has gcloud/gsutil pre-installed and authenticated
set -euo pipefail

export GCP_PROJECT_ID=groovy-rope-496901-d2
export GCS_BUCKET_NAME=predictive-maintenance-artifacts
export GCP_REGION=us-central1
export BILLING_ACCOUNT_ID=01A8CA-7630A8-16FA8E

echo "▶ Setting active project..."
gcloud config set project $GCP_PROJECT_ID

echo "▶ Enabling required APIs (this takes ~1 min)..."
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  compute.googleapis.com

echo "▶ Creating GCS bucket..."
gsutil mb -l $GCP_REGION gs://$GCS_BUCKET_NAME || echo "  Bucket already exists — skipping"

echo "▶ Creating bucket folder structure..."
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/data/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/models/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/pipelines/.keep
gsutil cp /dev/null gs://$GCS_BUCKET_NAME/artifacts/.keep

echo "▶ Creating billing budget alert at \$30..."
gcloud billing budgets create \
  --billing-account=$BILLING_ACCOUNT_ID \
  --display-name="predictive-maintenance-budget" \
  --budget-amount=30USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9

echo ""
echo "✅ GCP init complete"
echo ""
echo "Verify with:"
echo "  gcloud services list --enabled | grep -E 'aiplatform|storage|artifactregistry|cloudbuild|run|compute'"
echo "  gsutil ls gs://$GCS_BUCKET_NAME"
