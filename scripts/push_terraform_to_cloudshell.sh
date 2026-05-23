#!/usr/bin/env bash
# Pushes local terraform/ files to Google Cloud Shell.
# Run this from your Mac terminal — Cloud Shell must be active in a browser tab.
#
# Usage:
#   bash scripts/push_terraform_to_cloudshell.sh

set -euo pipefail

LOCAL_TF_DIR="$(cd "$(dirname "$0")/.." && pwd)/terraform"
REMOTE_DIR="predictive-maintenance-vertex/terraform"

echo "==> Local terraform dir : $LOCAL_TF_DIR"
echo "==> Remote Cloud Shell  : cloudshell:$REMOTE_DIR"
echo ""

# Ensure the remote directory exists
gcloud alpha cloud-shell ssh --command="mkdir -p $REMOTE_DIR" --quiet

# Upload every .tf file
for filepath in "$LOCAL_TF_DIR"/*.tf; do
    filename="$(basename "$filepath")"
    echo "  Uploading $filename ..."
    gcloud alpha cloud-shell scp \
        "localhost:$filepath" \
        "cloudshell:$REMOTE_DIR/$filename"
done

echo ""
echo "==> Done. All .tf files pushed."
echo ""
echo "Next steps — paste into Cloud Shell:"
echo "  cd ~/predictive-maintenance-vertex/terraform"
echo "  terraform init"
echo "  terraform plan \\"
echo "    -var=\"project_id=groovy-rope-496901-d2\" \\"
echo "    -var=\"bucket_name=predictive-maintenance-artifacts\""
