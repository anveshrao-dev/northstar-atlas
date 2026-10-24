"""Contained temporary checkout data; never copies credentials or runtime stores."""
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def checkout(test):
    temporary = tempfile.TemporaryDirectory(dir=ROOT / '.local/tmp')
    test.addCleanup(temporary.cleanup)
    root = Path(temporary.name)
    for relative in ('pyproject.toml', 'src/northstar_atlas/__init__.py',
                     'src/northstar_atlas/ui/app.py', 'data/corpus_manifest.json'):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    shutil.copytree(ROOT / 'data/policies', root / 'data/policies')
    return root
