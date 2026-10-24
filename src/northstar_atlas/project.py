"""Trusted process configuration and read-only checkout validation.

NORTHSTAR_PROJECT_ROOT is operator configuration, never a request/UI parameter
and never loaded from .env. Without it, only this package's source checkout is
eligible. These guards are not an operating-system security boundary.
"""
import os
from pathlib import Path
import tomllib

from northstar_atlas.ingestion.document_loader import load_documents


class ProjectRootError(ValueError):
    """Fixed safe configuration errors; no paths or source contents in messages."""


def _root_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ProjectRootError('project root must be an explicit absolute path')
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_dir():
        raise ProjectRootError('project root must be an existing directory without symlink aliases')
    return resolved


def contained_path(root: Path, relative: str) -> Path:
    """Validate a local path without creating it; reject all symlink components."""
    try:
        root = _root_path(root)
        part = Path(relative)
        if part.is_absolute() or '..' in part.parts:
            raise ValueError
        path = root / part
        if not path.resolve().is_relative_to(root):
            raise ValueError
        for component in (path, *path.parents):
            if component == root:
                break
            if component.is_symlink():
                raise ValueError
        return path
    except (OSError, RuntimeError, TypeError, ValueError):
        raise ProjectRootError('project path is invalid or escapes the authorized checkout') from None


def validate_project_root(candidate: Path | None = None) -> Path:
    """Resolve authority independently of candidate, then verify the entire corpus.

An explicit candidate must equal the operator-selected root. Merely containing a
valid manifest does not let a caller redirect an already configured service.
No ancestor search, cwd fallback, directories, credentials or providers are used.
"""
    try:
        configured = os.environ.get('NORTHSTAR_PROJECT_ROOT')
        authority = configured if configured is not None else Path(__file__).absolute().parents[2]
        root = _root_path(authority)
        if candidate is not None and _root_path(candidate) != root:
            raise ValueError
        project = contained_path(root, 'pyproject.toml')
        if tomllib.loads(project.read_text())['project']['name'] != 'northstar-atlas':
            raise ValueError
        if not contained_path(root, 'src/northstar_atlas/__init__.py').is_file():
            raise ValueError
        for relative in ('.local', '.venv', '.env', 'data/corpus_manifest.json', 'data/policies'):
            contained_path(root, relative)
        # Reuse the canonical loader: verify all hashes, metadata and statuses,
        # including excluded drafts. Never rewrite a source or its manifest.
        load_documents(root, mode='draft_review')
        return root
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        raise ProjectRootError('invalid or unauthorized project root/corpus; no resources opened') from None
