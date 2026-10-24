"""Exact persisted vectors and CLI separation, without cloud access."""

import json
from pathlib import Path
import tempfile
import unittest

from test_vector_store import ROOT, AS_OF, FakeProvider
from northstar_atlas.embeddings import embed_chunks
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.vector_store.preparation import freeze_snapshot
from northstar_atlas.vector_store.records import IndexConfig
from northstar_atlas.vector_store.pinecone_cloud import CloudError, CloudSettings, sdk_client
from northstar_atlas.vector_store.pinecone_workflow import save_snapshot, load_snapshot


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        provider = FakeProvider()
        chunk = chunk_document(load_documents(ROOT, reference_date=AS_OF)[0]).chunks[0]
        self.snapshot = freeze_snapshot(embed_chunks([chunk], provider), IndexConfig(provider.model, 256), "test-corpus")

    def test_frozen_snapshot_roundtrip_and_repeated_save(self):
        path = save_snapshot(self.root, self.snapshot)
        raw = path.read_bytes()
        loaded = load_snapshot(self.root, self.snapshot.namespace)
        self.assertEqual(loaded, self.snapshot)
        self.assertEqual(loaded.snapshot_id, self.snapshot.snapshot_id)
        save_snapshot(self.root, self.snapshot)
        self.assertEqual(path.read_bytes(), raw)

    def test_tampering_and_invalid_namespace_fail_closed(self):
        path = save_snapshot(self.root, self.snapshot)
        data = json.loads(path.read_text()); data["records"][0]["text"] += " changed"
        path.write_text(json.dumps(data))
        with self.assertRaises(CloudError): load_snapshot(self.root, self.snapshot.namespace)
        with self.assertRaises(CloudError): load_snapshot(self.root, "../../other")
        with self.assertRaises(CloudError): save_snapshot(self.root, self.snapshot)

    def test_interrupted_snapshot_publication_leaves_no_partial_final(self):
        from unittest.mock import patch
        with patch("os.link", side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError): save_snapshot(self.root, self.snapshot)
        final = self.root / ".local/pinecone/snapshots" / (self.snapshot.snapshot_id + ".json")
        self.assertFalse(final.exists())
        save_snapshot(self.root, self.snapshot)
        self.assertEqual(load_snapshot(self.root, self.snapshot.namespace), self.snapshot)

    def test_status_before_index_and_after_partial_upload_is_read_only(self):
        from types import SimpleNamespace
        from dataclasses import asdict
        from test_pinecone_cloud import Indexes, description, KEY
        from northstar_atlas.vector_store.pinecone_cloud import record_confirmation
        from test_pinecone_cloud import cloud_settings, PROJECT_ID
        from northstar_atlas.vector_store.pinecone_workflow import status_report, RequestMeter
        from northstar_atlas.vector_store.pinecone_state import PublicationState
        from northstar_atlas.vector_store.pinecone_store import PineconeTarget
        (self.root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
        settings = cloud_settings()
        record_confirmation(self.root, settings, project_id=PROJECT_ID)
        api = Indexes(); client = SimpleNamespace(indexes=api)
        self.assertEqual(status_report(self.root, settings, client, RequestMeter())["state"], "not_created")
        api.value = description(False); api.value["host"] = None
        self.assertEqual(status_report(self.root, settings, client, RequestMeter())["state"], "initializing")
        api.value = description()
        self.assertEqual(status_report(self.root, settings, client, RequestMeter())["state"], "not_indexed")
        target = PineconeTarget(api.value["name"], api.value["host"], PROJECT_ID)
        state = PublicationState(self.root, asdict(target))
        with state.lock():
            state.write_run(self.snapshot.namespace, {"state":"incomplete", "phase":"uploading", "acknowledged":50})
        result = status_report(self.root, settings, client, RequestMeter())
        self.assertEqual(result["state"], "incomplete")
        self.assertEqual(result["attempts"][0]["acknowledged"], 50)
        self.assertFalse(result["remote_audited"])
        self.assertEqual(api.created, [])

    def test_sdk_construction_is_offline_explicit_and_secret_safe(self):
        import os
        import socket
        from test_pinecone_cloud import cloud_settings
        from unittest.mock import patch
        with patch.dict(os.environ, {"PINECONE_DEBUG_CURL":"true", "HTTPS_PROXY":"http://untrusted.invalid"}):
            with patch.object(socket, "create_connection", side_effect=AssertionError("network forbidden")):
                with sdk_client(cloud_settings()) as client:
                    self.assertNotIn("PINECONE_DEBUG_CURL", os.environ)
                    self.assertNotIn("HTTPS_PROXY", os.environ)
                    self.assertEqual(client._config.host, "https://api.pinecone.io")
            self.assertEqual(os.environ["PINECONE_DEBUG_CURL"], "true")


    def test_connect_uses_explicit_target_and_rejects_wrong_index_before_transport(self):
        from dataclasses import replace
        from types import SimpleNamespace
        from test_pinecone_cloud import cloud_settings, description
        from northstar_atlas.vector_store.pinecone_workflow import connect_target
        settings = replace(cloud_settings(), name='synthetic-alternative')
        described, opened = [], []
        value = description(); value['name'] = settings.name
        def describe(*, name):
            described.append(name)
            return value
        def index(*, host):
            opened.append(host)
            return object()
        client = SimpleNamespace(indexes=SimpleNamespace(describe=describe), index=index)
        meter = SimpleNamespace(attach=lambda index: None)
        target, _ = connect_target(client, meter, settings)
        self.assertEqual(target.name, 'synthetic-alternative')
        self.assertEqual(target.project_id, '00000000-0000-4000-8000-000000000001')
        self.assertEqual(described, ['synthetic-alternative'])
        self.assertEqual(opened, [value['host']])
        value['name'] = 'wrong-index'
        with self.assertRaises(CloudError):
            connect_target(client, meter, settings)
        self.assertEqual(len(opened), 1)

    def test_explicit_empty_namespace_never_uses_active_generation(self):
        from dataclasses import asdict
        from unittest.mock import patch
        from test_pinecone_cloud import cloud_settings, description
        from northstar_atlas.vector_store.pinecone_store import PineconeTarget
        from northstar_atlas.vector_store.pinecone_state import PublicationState
        from northstar_atlas.vector_store.pinecone_workflow import active_snapshot
        settings = cloud_settings()
        target = PineconeTarget(settings.name, description()['host'], settings.project_id)
        (self.root / 'pyproject.toml').write_text('[project]\nname="northstar-atlas"\n')
        save_snapshot(self.root, self.snapshot)
        state = PublicationState(self.root, asdict(target))
        with state.lock():
            state.write_run(self.snapshot.namespace, {'state': 'ready'})
            state.publish(self.snapshot.namespace)
        # The fixture has fake vectors; corpus compatibility is independently tested.
        with patch('northstar_atlas.vector_store.pinecone_workflow.validate_m33') as source_check:
            with self.assertRaises(CloudError):
                active_snapshot(self.root, target, '')
            source_check.assert_not_called()


if __name__ == "__main__": unittest.main()
