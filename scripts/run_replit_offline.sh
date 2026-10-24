#!/usr/bin/env bash
# Hosted authored demo only; this command never installs packages.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [ "$(pwd -P)" != "$root" ]; then
  echo "Run from the Northstar Atlas checkout root" >&2
  exit 1
fi
export NORTHSTAR_PROJECT_ROOT="$root"
export NORTHSTAR_UI_ENABLE_LIVE=0
export NORTHSTAR_UI_ENABLE_PINECONE=0
unset NEBIUS_API_KEY PINECONE_API_KEY FIREWORKS_API_KEY
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
preview_host="${REPLIT_DEV_DOMAIN:?Replit editor preview hostname is required}"
case "$preview_host" in
  *[!a-zA-Z0-9.-]*|.*) echo "Invalid preview hostname" >&2; exit 1 ;;
esac
case "$preview_host" in
  *.replit.dev) ;;
  *) echo "Expected a Replit development hostname" >&2; exit 1 ;;
esac
.venv/bin/python -B - <<'CHECK'
from pathlib import Path
import sys
from northstar_atlas.project import contained_path, validate_project_root
if sys.version_info[:2] != (3, 12):
    raise SystemExit('The preview requires Python 3.12')
root = validate_project_root(Path.cwd())
for relative in ('.local/tmp', '.local/cache', '.venv/bin'):
    contained_path(root, relative)
CHECK
mkdir -p .local/tmp .local/cache
export TMPDIR="$root/.local/tmp"
export XDG_CACHE_HOME="$root/.local/cache"
exec .venv/bin/python -B -m streamlit run src/northstar_atlas/ui/app.py \
  --server.address=0.0.0.0 --server.port=8501 \
  --browser.serverAddress="$preview_host" --browser.serverPort=443 \
  --server.enableCORS=true --server.enableXsrfProtection=true \
  --server.headless=true --server.showEmailPrompt=false \
  --server.fileWatcherType=none --browser.gatherUsageStats=false \
  --client.showErrorDetails=none
