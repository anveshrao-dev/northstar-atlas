#!/usr/bin/env bash
# Separate explicit controlled-live startup; never installs or initializes a ledger.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(pwd -P)" = "$root" || { echo "Run from the Northstar Atlas checkout root" >&2; exit 1; }
export NORTHSTAR_PROJECT_ROOT="$root"
export NORTHSTAR_CONTROLLED_LIVE=1
export NORTHSTAR_UI_ENABLE_LIVE=0 NORTHSTAR_UI_ENABLE_PINECONE=0
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VECTOR_STORE="${VECTOR_STORE:-chroma}"
case "$VECTOR_STORE" in
  chroma|pinecone) ;;
  *) echo "Unsupported retrieval backend" >&2; exit 1 ;;
esac
.venv/bin/python -B - <<'CHECK'
from pathlib import Path
import sys
from northstar_atlas.project import contained_path, validate_project_root
if sys.version_info[:2] != (3, 12):
    raise SystemExit('Controlled preview requires Python 3.12')
root = validate_project_root(Path.cwd())
for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
    contained_path(root, relative)
CHECK
export TMPDIR="$root/.local/tmp" XDG_CACHE_HOME="$root/.local/cache"
# Standard startup verifies external access and the model catalog. When graph
# use is explicitly authorized, preflight stays local; external checks wait for
# a full workflow reservation on submit. No inference or ledger migration/setup.
.venv/bin/python -B -m northstar_atlas.live_preflight
export NORTHSTAR_UI_ENABLE_LIVE=1
if [ "$VECTOR_STORE" = pinecone ]; then
  export NORTHSTAR_UI_ENABLE_PINECONE=1
fi
# LangGraph stays off unless independently authorized through its server gate;
# each graph submission still needs the full durable reservation and preflight.
exec .venv/bin/python -B -m streamlit run src/northstar_atlas/ui/app.py \
  --server.address=0.0.0.0 --server.port=8501 \
  --browser.serverAddress="${REPLIT_DEV_DOMAIN:?Replit development hostname required}" --browser.serverPort=443 \
  --server.enableCORS=true --server.enableXsrfProtection=true \
  --server.headless=true --server.showEmailPrompt=false \
  --server.fileWatcherType=none --browser.gatherUsageStats=false \
  --client.showErrorDetails=none
