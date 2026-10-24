"""Durable local publication: incomplete attempts never become active."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from northstar_atlas.vector_store.pinecone_state import PublicationState
from northstar_atlas.vector_store.records import VectorStoreError

ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = "dev-" + "a" * 64


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="publication-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
        self.state = PublicationState(self.root, {"host": "offline.invalid", "project": "offline"})

    def test_incomplete_manifest_does_not_publish(self):
        with self.state.lock():
            self.state.write_run(NAMESPACE, {"state": "incomplete"})
            with self.assertRaises(VectorStoreError): self.state.publish(NAMESPACE)
            self.assertIsNone(self.state.active())

    def test_pointer_failure_preserves_previous_generation_and_can_resume(self):
        with self.state.lock():
            self.state.write_run(NAMESPACE, {"state": "ready"})
            self.state.publish(NAMESPACE)
            next_namespace = "dev-" + "b" * 64
            self.state.write_run(next_namespace, {"state": "ready"})
            import os
            replace = os.replace
            def fail_pointer(src, dest):
                if Path(dest).name == "active.json": raise OSError("interrupted publication")
                replace(src, dest)
            with patch("northstar_atlas.vector_store.pinecone_state.os.replace", side_effect=fail_pointer):
                with self.assertRaises(OSError): self.state.publish(next_namespace)
            self.assertEqual(self.state.active(), NAMESPACE)
            self.state.publish(next_namespace)
            self.assertEqual(self.state.active(), next_namespace)

    def test_corrupt_state_and_path_traversal_fail_closed(self):
        with self.state.lock():
            self.state.write_run(NAMESPACE, {"state": "ready"})
            for name in ("../outside", "dev-invalid", ""):
                with self.assertRaises(VectorStoreError): self.state.read_run(name)
            (self.state.path / "active.json").write_text('{"namespace":"../outside"}')
            with self.assertRaises(VectorStoreError): self.state.active()

    def test_second_writer_cannot_enter_lock(self):
        other = PublicationState(self.root, {"host": "offline.invalid", "project": "offline"})
        with self.state.lock():
            with self.assertRaisesRegex(VectorStoreError, "writer|lock"):
                with other.lock(): pass

