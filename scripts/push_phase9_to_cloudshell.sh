#!/usr/bin/env bash
# Pushes Phase 9 files to Google Cloud Shell.
# Run from your Mac terminal — Cloud Shell must be active in a browser tab.
#
# Usage:
#   bash scripts/push_phase9_to_cloudshell.sh

set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="predictive-maintenance-vertex"

echo "==> Pushing Phase 9 files to Cloud Shell"
echo ""

# Ensure remote directories exist
gcloud alpha cloud-shell ssh \
    --command="mkdir -p $REMOTE_ROOT/src/model $REMOTE_ROOT/src/data $REMOTE_ROOT/scripts $REMOTE_ROOT/model_artifacts" \
    --quiet

# ---- src/model source files (needed for local experiment runs) ----
echo "  [1/8] src/__init__.py"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/__init__.py" \
    "cloudshell:$REMOTE_ROOT/src/__init__.py"

echo "  [2/8] src/model/__init__.py"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/model/__init__.py" \
    "cloudshell:$REMOTE_ROOT/src/model/__init__.py"

echo "  [3/8] src/model/lstm.py"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/model/lstm.py" \
    "cloudshell:$REMOTE_ROOT/src/model/lstm.py"

echo "  [4/8] src/model/train.py  (updated — Vertex AI Experiments)"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/model/train.py" \
    "cloudshell:$REMOTE_ROOT/src/model/train.py"

echo "  [5/8] src/model/evaluate.py"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/model/evaluate.py" \
    "cloudshell:$REMOTE_ROOT/src/model/evaluate.py"

# ---- script + artifact ----
echo "  [6/8] scripts/run_experiments.py  (new)"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/scripts/run_experiments.py" \
    "cloudshell:$REMOTE_ROOT/scripts/run_experiments.py"

echo "  [7/8] model_artifacts/experiment_summary.json  (template)"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/model_artifacts/experiment_summary.json" \
    "cloudshell:$REMOTE_ROOT/model_artifacts/experiment_summary.json"

echo "  [8/8] src/data/__init__.py"
gcloud alpha cloud-shell scp \
    "localhost:$LOCAL_ROOT/src/data/__init__.py" \
    "cloudshell:$REMOTE_ROOT/src/data/__init__.py"

echo ""
echo "==> Done. Verifying files in Cloud Shell ..."
gcloud alpha cloud-shell ssh \
    --command="find $REMOTE_ROOT/src/model -name '*.py' | sort" \
    --quiet

echo ""
echo "==> All Phase 9 files pushed."
echo ""
echo "Next — paste into Cloud Shell:"
echo "  cd ~/predictive-maintenance-vertex"
echo "  python scripts/run_experiments.py --vertex-experiments"
