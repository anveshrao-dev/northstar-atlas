#!/usr/bin/env bash
# Explicit fresh-checkout installation, never called by the Replit Run button.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [ "$(pwd -P)" != "$root" ]; then
  echo "Run from the Northstar Atlas checkout root" >&2
  exit 1
fi
if [ -e .venv ] || [ -L .venv ]; then
  echo "Use a fresh preview checkout; existing .venv will not be replaced" >&2
  exit 1
fi
python3.12 -B - <<'CHECK'
from pathlib import Path
import sys
import tomllib
if sys.version_info[:2] != (3, 12):
    raise SystemExit('The preview requires Python 3.12')
root = Path.cwd().resolve()
paths = ('pyproject.toml', 'uv.lock', 'src/northstar_atlas/__init__.py',
         'data/corpus_manifest.json', '.local/tmp', '.local/replit-preview',
         '.local/replit-preview/ui-requirements.txt', '.local/uv-cache',
         '.local/uv-python', '.local/bin', '.local/pycache')
for relative in paths:
    path = root / relative
    if not path.resolve().is_relative_to(root) or any(
        p.is_symlink() for p in (path, *path.parents) if p != root and root in p.parents
    ):
        raise SystemExit('Preview setup path is a symlink or escapes the checkout')
for relative in paths[:4]:
    if not (root / relative).is_file():
        raise SystemExit('Preview setup requires the complete project checkout')
if tomllib.loads((root / 'pyproject.toml').read_text())['project']['name'] != 'northstar-atlas':
    raise SystemExit('Invalid project checkout')
CHECK
mkdir -p .local/tmp .local/replit-preview
export UV_CACHE_DIR="$root/.local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$root/.local/uv-python"
export UV_PYTHON_BIN_DIR="$root/.local/bin"
export UV_PROJECT_ENVIRONMENT="$root/.venv"
export UV_PYTHON_DOWNLOADS=never
export TMPDIR="$root/.local/tmp"
export PYTHONPYCACHEPREFIX="$root/.local/pycache"
# Match the whole output: exact pin, optional platform target, one line only.
# pipefail also rejects a failed uv command even if it prints a valid version.
if ! uv --version | python3.12 -B -c '
import re
import sys
target = r"[a-z][a-z0-9_]*(?:-[a-z][a-z0-9_]*){2,3}"
pattern = rf"uv 0\.12\.23(?: \({target}\))?(?:\r?\n)?"
sys.exit(0 if re.fullmatch(pattern, sys.stdin.read()) else 1)
'; then
  echo "Preview setup requires reviewed uv 0.12.23; no installation attempted" >&2
  exit 1
fi
uv export --locked --offline --extra ui --no-emit-project \
  --prune chromadb --prune sentence-transformers --prune torch --prune transformers \
  --no-header --no-annotate \
  --output-file .local/replit-preview/ui-requirements.txt
python3.12 -B - <<'CHECK'
from hashlib import sha256
from pathlib import Path
# Fingerprint of the reviewed 35-package export, including versions and hashes.
# A different export requires review, not an automatic new dependency selection.
expected = '0d7a42c564358febb8bde9167fc108c7e022953946cd9fef5e8d1ec38f1e1faf'
actual = sha256(Path('.local/replit-preview/ui-requirements.txt').read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit('Export differs from the reviewed UI subset; stop for dependency review')
CHECK
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python --require-hashes --only-binary :all: \
  -r .local/replit-preview/ui-requirements.txt
uv pip install --python .venv/bin/python --no-deps --editable .
export NORTHSTAR_PROJECT_ROOT="$root"
.venv/bin/python -B -c 'from pathlib import Path; from northstar_atlas.project import validate_project_root; validate_project_root(Path.cwd())'
echo "Offline UI subset installed. Use the Replit Run button; do not run uv sync here."
