"""Offline SDK boundary tests: real Documents API, in-memory HTTP responses."""

from copy import deepcopy
from dataclasses import replace
from datetime import date
import importlib.util
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from test_vector_store import FakeProvider, ROOT, AS_OF
from northstar_atlas.embeddings import embed_chunks
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.refiner import refine_chunks
from northstar_atlas.vector_store.records import IndexConfig, VectorStoreError
from northstar_atlas.vector_store.preparation import freeze_snapshot
from northstar_atlas.vector_store.pinecone_store import PineconeStore, PineconeTarget


def matches(record, expression):
    if "$and" in expression:
        return all(matches(record, x) for x in expression["$and"])
    for key, operations in expression.items():
        for op, value in operations.items():
            actual = record.get(key)
            if op == "$eq" and actual != value: return False
            if op == "$in" and actual not in value: return False
            if op == "$gt" and not actual > value: return False
            if op == "$lte" and not actual <= value: return False
    return True


class MemoryHTTP:
    """Only the HTTP boundary is replaced; SDK codecs/pagination remain real."""

    def __init__(self):
        self.spaces = {}
        self.calls = []
        self.fail = None
        self.visibility_delay = 0
        self.score = 0.75
        self.ignore_filter = False
        self.float_metadata = False

    def post(self, path, *, timeout, json):
        import httpx
        self.calls.append((path, deepcopy(json), timeout))
        _, _, namespace, _, operation = path.split("/")
        records = self.spaces.setdefault(namespace, {})
        if self.fail == operation:
            raise RuntimeError("private provider error including SECRET")
        if operation == "upsert":
            for record in json["documents"]:
                records[record["_id"]] = deepcopy(record)
            if self.fail == "timeout_after_success":
                raise TimeoutError("SECRET")
            body = {"upserted_count": 0 if self.fail == "partial" else len(json["documents"])}
        elif operation == "list":
            ids = sorted(records)
            offset = 0 if self.fail == "empty_pages" else int(json.get("pagination_token", "0"))
            page = ids[offset:offset + 2]
            body = {"namespace": namespace, "documents": [{"_id": i} for i in page]}
            if offset + 2 < len(ids): body["pagination"] = {"next": str(offset + 2)}
            if self.fail == "empty_pages":
                if len(self.calls) > 4: raise RuntimeError("test watchdog stopped unbounded pagination")
                body = {"namespace": namespace, "documents": [], "pagination": {"next": "repeat"}}
        elif operation == "fetch":
            body = {"namespace": namespace, "documents": {i: deepcopy(records[i])
                    for i in json["ids"] if i in records}}
            if self.float_metadata:
                for record in body["documents"].values():
                    for key in ("effective_ordinal", "embedding_dimension", "storage_schema_version"):
                        record[key] = float(record[key])
        elif operation == "search":
            assert json["score_by"][0]["type"] == "dense_vector"
            assert json["score_by"][0]["field"] == "embedding"
            found = []
            if self.visibility_delay:
                self.visibility_delay -= 1
            else:
                for record in records.values():
                    if self.ignore_filter or matches(record, json["filter"]):
                        found.append({"_id": record["_id"], "_score": self.score,
                                      **{k: record[k] for k in json["include_fields"] if k in record}})
            body = {"namespace": namespace, "matches": found[:json["top_k"]]}
        else:
            raise AssertionError(f"Unexpected operation: {operation}")
        if operation in ("list", "fetch", "search"):
            body["usage"] = {"read_units": 1}
        return httpx.Response(200, json=body)


@unittest.skipUnless(importlib.util.find_spec("pinecone"), "optional Pinecone extra not installed")
class PineconeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        cls.hr, cls.it = [chunk_document(d).chunks[0] for d in cls.docs[:2]]

    def setUp(self):
        from pinecone.client.documents import Documents
        self.no_network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="pinecone-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
        self.provider = FakeProvider()
        self.config = IndexConfig(self.provider.model, 256)
        self.snapshot = freeze_snapshot(embed_chunks([self.hr, self.it], self.provider), self.config, "corpus-1")
        self.http = MemoryHTTP()
        self.documents = Documents(http=self.http, host="https://offline.invalid")
        self.target = PineconeTarget("northstar-atlas-dev", "https://offline.invalid", "offline-project")
        self.description = {"name": self.target.name, "host": self.target.host,
                            "status": {"ready": True}, "deletion_protection": "enabled",
                            "deployment": {"deployment_type": "managed", "cloud": "azure", "region": "eastus2"},
                            "schema": {"fields": {"embedding": {"type": "dense_vector", "dimension": 3, "metric": "cosine"}}}}
        self.current = "corpus-1"
        self.store = self.make_store()

    def make_store(self, snapshot=None):
        return PineconeStore(self.root, snapshot or self.snapshot, self.target,
                             documents=self.documents, describe_target=lambda: deepcopy(self.description),
                             current_fingerprint=lambda: self.current, poll_attempts=3, poll_interval=0)

    def search(self, **kwargs):
        return self.store.search("access approval", self.provider, corpus_fingerprint=self.current,
                                 reference_date=AS_OF, **kwargs)

    def test_publish_reopen_and_repeat_skip_upsert(self):
        report = self.store.synchronize()
        self.assertEqual((report.total, report.upserted, report.reused), (2, 2, False))
        self.assertEqual(self.store.status()["state"], "ready")
        self.store = self.make_store()
        again = self.store.synchronize()
        self.assertEqual((again.total, again.upserted, again.reused), (2, 0, True))
        self.assertEqual(sum(p.endswith("upsert") for p, _, _ in self.http.calls), 1)
        hits = self.search(mode="draft_review")
        self.assertEqual({h.chunk for h in hits}, {self.hr, self.it})
        self.assertTrue(all(h.cosine_distance == 0.25 for h in hits))

    def test_snapshot_identity_changes_with_sources_vectors_and_config(self):
        a = freeze_snapshot(embed_chunks([self.it, self.hr], self.provider), self.config, "corpus-1")
        self.assertEqual(a.snapshot_id, self.snapshot.snapshot_id)
        changed = replace(self.hr, chunk_id="changed", content_hash="a" * 64)
        b = freeze_snapshot(embed_chunks([changed, self.it], self.provider), self.config, "corpus-2")
        self.assertNotEqual(a.snapshot_id, b.snapshot_id)
        self.assertTrue(a.namespace.startswith("dev-"))
        self.assertEqual(len(a.namespace), 68)

    def test_new_generation_keeps_old_records_out_of_active_search(self):
        self.store.synchronize()
        new = replace(self.hr, chunk_id="new", content_hash="b" * 64)
        snap = freeze_snapshot(embed_chunks([new], self.provider), self.config, "corpus-2")
        self.current = "corpus-2"
        self.store = self.make_store(snap)
        self.store.synchronize()
        self.assertEqual([h.chunk.chunk_id for h in self.search()], ["new"])
        self.assertEqual(len(self.http.spaces[self.snapshot.namespace]), 2)

    def test_approval_filter_runs_before_ranking_and_dates_are_enforced(self):
        future = replace(self.hr, chunk_id="future", effective_date=date(2027, 1, 1))
        self.store = self.make_store(freeze_snapshot(embed_chunks([self.it, future, self.hr], self.provider),
                                                   self.config, "corpus-1"))
        self.store.synchronize()
        self.assertEqual([h.chunk for h in self.search()], [self.hr])
        self.assertEqual(len(self.search(mode="draft_review")), 3)
        self.http.ignore_filter = True
        with self.assertRaisesRegex(VectorStoreError, "eligib|authority"):
            self.search()

    def test_partial_acknowledgment_blocks_search_and_resumes(self):
        self.http.fail = "partial"
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.assertEqual(self.store.status()["state"], "incomplete")
        with self.assertRaises(VectorStoreError): self.search()
        self.http.fail = None
        self.store.synchronize()
        self.assertEqual(len(self.search(mode="draft_review")), 2)

    def test_timeout_after_success_sanitizes_error_and_reconciles_fixed_ids(self):
        self.http.fail = "timeout_after_success"
        with self.assertRaises(VectorStoreError) as error: self.store.synchronize()
        self.assertNotIn("SECRET", str(error.exception))
        self.assertEqual(sum(p.endswith("upsert") for p, _, _ in self.http.calls), 1)
        self.http.fail = None
        self.store.synchronize()
        self.assertEqual(len(self.http.spaces[self.snapshot.namespace]), 2)

    def test_delayed_visibility_is_polled_and_timeout_is_incomplete(self):
        self.http.visibility_delay = 1
        self.store.synchronize()
        self.assertEqual(self.store.status()["state"], "ready")
        self.http.visibility_delay = 999
        with self.assertRaisesRegex(VectorStoreError, "visibility"):
            self.store.synchronize()
        with self.assertRaises(VectorStoreError): self.search()

    def test_unknown_namespace_and_extra_ids_are_never_overwritten(self):
        self.http.spaces[self.snapshot.namespace] = {"alien": {"_id": "alien"}}
        with self.assertRaisesRegex(VectorStoreError, "owned|ownership"):
            self.store.synchronize()
        self.assertFalse(any(p.endswith("upsert") for p, _, _ in self.http.calls))

    def test_full_audit_rejects_tampered_text_vectors_and_extra_ids(self):
        self.store.synchronize()
        original = deepcopy(self.http.spaces[self.snapshot.namespace])
        for change in ("text", "embedding", "extra"):
            with self.subTest(change=change):
                self.http.spaces[self.snapshot.namespace] = deepcopy(original)
                records = self.http.spaces[self.snapshot.namespace]
                if change == "extra": records["alien"] = {"_id": "alien"}
                elif change == "text": records[self.hr.chunk_id]["text"] += "altered"
                else: records[self.hr.chunk_id]["embedding"] = [1.0, 0.0, 0.0]
                with self.assertRaises(VectorStoreError): self.store.synchronize()

    def test_integral_metadata_round_trip_and_non_integral_rejection(self):
        self.http.float_metadata = True
        self.store.synchronize()
        self.assertEqual(len(self.search(mode="draft_review")), 2)
        self.http.spaces[self.snapshot.namespace][self.hr.chunk_id]["embedding_dimension"] = 3.5
        with self.assertRaises(VectorStoreError): self.store.synchronize()

    def test_invalid_query_mode_top_k_model_and_score_fail_closed(self):
        self.store.synchronize()
        for query in ("", " ", None):
            with self.assertRaises(ValueError):
                self.store.search(query, self.provider, corpus_fingerprint=self.current)
        for options in ({"mode": "all"}, {"top_k": 0}, {"top_k": True}):
            with self.assertRaises(ValueError): self.search(**options)
        other = FakeProvider()
        other.model = replace(other.model, revision="wrong")
        with self.assertRaises(VectorStoreError):
            self.store.search("query", other, corpus_fingerprint=self.current)
        for score in (float("nan"), 1.1, -1.1, True):
            self.http.score = 0.75
            self.store.synchronize()
            self.http.score = score
            with self.assertRaises(VectorStoreError): self.search()

    def test_stale_corpus_and_incompatible_target_block_before_writes(self):
        self.current = "changed"
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.current = "corpus-1"
        for field, value in (("cloud", "aws"), ("region", "wrong")):
            original = self.description["deployment"][field]
            self.description["deployment"][field] = value
            with self.assertRaises(VectorStoreError): self.store.synchronize()
            self.description["deployment"][field] = original
        self.description["schema"]["fields"]["embedding"]["dimension"] = 384
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.assertFalse(any(p.endswith("upsert") for p, _, _ in self.http.calls))

    def test_siblings_preserve_sections_and_missing_sibling_fails(self):
        provider = FakeProvider()
        provider.max_input_tokens = 200
        chunks = chunk_document(next(d for d in self.docs if d.document_id == "PLAT-001")).chunks
        family = [c for c in chunks if c.section_id == "6.1"]
        refined = refine_chunks(family, provider).require_complete()
        self.assertGreater(len(refined), 1)
        config = IndexConfig(provider.model, provider.max_input_tokens)
        self.provider = provider
        snapshot = freeze_snapshot(embed_chunks(refined, provider), config, "corpus-1")
        self.store = self.make_store(snapshot)
        self.store.synchronize()
        hit = self.search(mode="draft_review", top_k=1)[0]
        self.assertEqual(len(hit.context), len(refined) - 1)
        self.assertTrue(all(c.section_id == "6.1" for c in hit.context))
        del self.http.spaces[snapshot.namespace][hit.context[0].chunk_id]
        with self.assertRaisesRegex(VectorStoreError, "sibling|missing"):
            self.search(mode="draft_review", top_k=1)
        self.assertEqual(self.store.status()["state"], "incomplete")
        self.assertFalse(self.store._audited)

    def test_untrusted_source_change_during_upload_never_publishes(self):
        original = self.http.post
        def change_after_upload(path, **kwargs):
            response = original(path, **kwargs)
            if path.endswith("upsert"): self.current = "changed"
            return response
        self.http.post = change_after_upload
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.assertEqual(self.store.status()["state"], "incomplete")

    def test_payload_budget_rejects_oversized_metadata_without_truncation(self):
        huge = replace(self.hr, text="x" * 50_000)
        snapshot = freeze_snapshot(embed_chunks([huge], self.provider), self.config, "corpus-1")
        with self.assertRaisesRegex(VectorStoreError, "metadata|budget"):
            self.make_store(snapshot).synchronize()
        self.assertEqual(self.http.calls, [])

    def test_state_symlink_cannot_escape_project(self):
        (self.root / ".local").mkdir()
        (self.root / ".local/pinecone").symlink_to(ROOT / ".local")
        with self.assertRaisesRegex(VectorStoreError, "symlink|outside"):
            self.store.synchronize()

    def test_failed_audit_invalidates_readiness_until_reconciled(self):
        self.store.synchronize()
        del self.http.spaces[self.snapshot.namespace][self.it.chunk_id]
        with self.assertRaises(VectorStoreError): self.store.audit()
        self.assertEqual(self.store.status()["state"], "incomplete")
        with self.assertRaises(VectorStoreError): self.search()

    def test_documents_transport_must_match_selected_host_before_any_operation(self):
        self.documents._host = "https://another-target.invalid"
        with self.assertRaisesRegex(VectorStoreError, "host|transport"):
            self.store.synchronize()
        self.assertEqual(self.http.calls, [])

    def test_repeated_empty_pagination_is_bounded_before_upsert(self):
        self.http.fail = "empty_pages"
        with self.assertRaises(VectorStoreError): self.store.synchronize()
        self.assertLessEqual(len(self.http.calls), 2)
        self.assertFalse(any(path.endswith("upsert") for path, _, _ in self.http.calls))

    def test_reopening_a_ready_snapshot_requires_startup_audit(self):
        self.store.synchronize()
        del self.http.spaces[self.snapshot.namespace][self.it.chunk_id]
        self.store = self.make_store()
        with self.assertRaises(VectorStoreError): self.search()

    def test_batches_never_exceed_count_and_partial_update_retains_old_pointer(self):
        self.store.synchronize()
        chunks = [replace(self.hr, chunk_id=f"batch-{i:03d}") for i in range(101)]
        self.current = "corpus-2"
        snap = freeze_snapshot(embed_chunks(chunks, self.provider), self.config, self.current)
        new = self.make_store(snap)
        self.http.fail = "partial"
        with self.assertRaises(VectorStoreError): new.synchronize()
        self.assertEqual(new.state.active(), self.snapshot.namespace)
        self.http.fail = None
        new.synchronize()
        writes = [body for path, body, _ in self.http.calls if path.endswith("upsert")]
        self.assertTrue(all(len(body["documents"]) <= 50 for body in writes))
        self.assertEqual(len(self.http.spaces[snap.namespace]), 101)

    def test_cleanup_requires_owned_inactive_exact_target_and_reports_failure(self):
        self.store.synchronize()
        self.current = "corpus-2"
        snap = freeze_snapshot(embed_chunks([self.hr], self.provider), self.config, self.current)
        new = self.make_store(snap)
        new.synchronize()
        new.delete_namespace = lambda namespace: self.http.spaces.pop(namespace, None)
        with self.assertRaises(VectorStoreError): new.cleanup(new.namespace, approved_namespace=new.namespace)
        with self.assertRaises(VectorStoreError): new.cleanup(self.snapshot.namespace, approved_namespace="wrong")
        def fail(namespace): raise RuntimeError("SECRET")
        new.delete_namespace = fail
        with self.assertRaises(VectorStoreError) as error:
            new.cleanup(self.snapshot.namespace, approved_namespace=self.snapshot.namespace)
        self.assertNotIn("SECRET", str(error.exception))
        self.assertEqual(new.status()["state"], "ready")
        new.delete_namespace = lambda namespace: self.http.spaces.pop(namespace, None)
        new.cleanup(self.snapshot.namespace, approved_namespace=self.snapshot.namespace)
        self.assertEqual(new.state.read_run(self.snapshot.namespace)["state"], "deleted")

    def test_snapshot_rejects_duplicate_ids_invalid_vectors_and_incomplete_siblings(self):
        records = embed_chunks([self.hr, self.it], self.provider)
        for broken in ([records[0], records[0]], [replace(records[0], vector=(0., 0., 0.))],
                       [replace(records[0], vector=(1., 0.))],
                       [replace(records[0], vector=(2., 0., 0.))]):
            with self.assertRaises(ValueError): freeze_snapshot(broken, self.config, "corpus-1")
        changed = [replace(records[0], vector=(1., 0., 0.)), records[1]]
        self.assertNotEqual(freeze_snapshot(changed, self.config, "corpus-1").snapshot_id,
                            self.snapshot.snapshot_id)


@unittest.skipUnless(os.environ.get("NORTHSTAR_RUN_MODEL_TESTS") == "1" and importlib.util.find_spec("pinecone"),
                     "enable cached MiniLM and install optional SDK for offline corpus integration")
class RealCorpusOfflineTests(unittest.TestCase):
    def test_all_189_cached_model_records_round_trip_without_network(self):
        from pinecone.client.documents import Documents
        from northstar_atlas.local_embeddings import MiniLMProvider
        from northstar_atlas.vector_store.preparation import prepare_corpus, corpus_snapshot
        provider = MiniLMProvider(ROOT)
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            prepared = prepare_corpus(ROOT, provider)
            self.assertEqual((prepared.documents, prepared.canonical_chunks, len(prepared.records)), (6, 151, 189))
            self.assertLessEqual(prepared.max_tokens, 256)
            snapshot = freeze_snapshot(prepared.records, prepared.config, prepared.fingerprint)
            http = MemoryHTTP()
            documents = Documents(http=http, host="https://offline.invalid")
            target = PineconeTarget("northstar-atlas-dev", "https://offline.invalid", "offline-project")
            description = {"name": target.name, "host": target.host, "status": {"ready": True},
                           "deletion_protection": "enabled",
                           "deployment": {"deployment_type": "managed", "cloud": "azure", "region": "eastus2"},
                           "schema": {"fields": {"embedding": {"type": "dense_vector", "dimension": 384, "metric": "cosine"}}}}
            with tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="pinecone-corpus-") as temp:
                root = Path(temp)
                (root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
                store = PineconeStore(root, snapshot, target, documents=documents,
                                      describe_target=lambda: description,
                                      current_fingerprint=lambda: corpus_snapshot(ROOT)[1], poll_interval=0)
                self.assertEqual(store.synchronize().total, 189)
                self.assertTrue(store.synchronize().reused)
                store.audit()
                for key, wire in http.spaces[snapshot.namespace].items():
                    self.assertEqual(wire["_id"], key)
                    self.assertEqual(len(wire["embedding"]), 384)
                for mode in ("approved_only", "draft_review"):
                    hits = store.search("Who approves a Terraform production plan?", provider,
                                        corpus_fingerprint=prepared.fingerprint, mode=mode, reference_date=AS_OF)
                    self.assertTrue(hits)
                    if mode == "approved_only":
                        self.assertTrue(all(h.chunk.document_id == "HR-001" for h in hits))

