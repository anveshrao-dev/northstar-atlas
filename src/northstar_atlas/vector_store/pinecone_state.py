"""Local, target-scoped publication manifests; no distributed-lock claim."""

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import tomllib

from .records import VectorStoreError, canonical_json, digest


class PublicationState:
    def __init__(self, root: Path, target: dict):
        self.root = Path(root).resolve(strict=True)
        project = self.root / "pyproject.toml"
        if not project.resolve().is_relative_to(self.root):
            raise VectorStoreError("project configuration resolves outside root")
        if tomllib.loads(project.read_text())["project"]["name"] != "northstar-atlas":
            raise VectorStoreError("expected Northstar Atlas project")
        self.path = self.root / ".local/pinecone" / digest(target)

    def check_paths(self) -> None:
        for path in (self.root / ".local", self.path.parent, self.path):
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise VectorStoreError("publication path is a symlink or outside project")
        if self.path.exists() and any(p.is_symlink() for p in self.path.rglob("*")):
            raise VectorStoreError("publication state contains a symlink")

    @contextmanager
    def lock(self):
        self.check_paths()
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "writer.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise VectorStoreError("another local writer holds the publication lock") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @staticmethod
    def validate_namespace(namespace: str) -> None:
        if not isinstance(namespace, str) or not re.fullmatch(r"dev-[0-9a-f]{64}", namespace):
            raise VectorStoreError("invalid owned namespace")

    def _read(self, name: str) -> dict | None:
        self.check_paths()
        try:
            data = json.loads((self.path / name).read_text())
            if not isinstance(data, dict): raise ValueError
            return data
        except FileNotFoundError:
            return None
        except (ValueError, UnicodeError):
            raise VectorStoreError("invalid publication state; explicit recovery required") from None

    def _write(self, name: str, data: dict) -> None:
        self.check_paths()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path,
                                             prefix="publication-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(canonical_json(data) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path / name)
            directory = os.open(self.path, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    def read_run(self, namespace: str) -> dict | None:
        self.validate_namespace(namespace)
        return self._read(namespace + ".json")

    def write_run(self, namespace: str, data: dict) -> None:
        self.validate_namespace(namespace)
        self._write(namespace + ".json", data)

    def active(self) -> str | None:
        data = self._read("active.json")
        if data is None: return None
        namespace = data.get("namespace")
        self.validate_namespace(namespace)
        return namespace

    def publish(self, namespace: str) -> None:
        run = self.read_run(namespace)
        if run is None or run.get("state") != "ready":
            raise VectorStoreError("cannot publish an incomplete generation")
        self._write("active.json", {"namespace": namespace})

    def runs(self) -> dict[str, dict]:
        self.check_paths()
        return {p.stem: self.read_run(p.stem) for p in self.path.glob("dev-*.json")}
