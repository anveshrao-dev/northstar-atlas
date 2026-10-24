"""Real local Chroma, deterministic fake vectors, and opt-in cached MiniLM."""

from dataclasses import replace
from datetime import date
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from northstar_atlas.embeddings import EmbeddingModel, EmbeddedChunk, embed_chunks
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.refiner import refine_chunks


ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 22)


class FakeProvider:
    model = EmbeddingModel("fake-test-only", "revision-1", 3, "l2")
    max_input_tokens = 256

    def token_counts(self, texts):
        return [len(text.split()) + 2 for text in texts]

    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            raw = [float(x + 1) for x in hashlib.sha256(text.encode()).digest()[:3]]
            norm = math.hypot(*raw)
            vectors.append([x / norm for x in raw])
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class StoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        cls.hr = chunk_document(docs[0]).chunks[0]
        cls.it = chunk_document(docs[1]).chunks[0]

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("northstar_atlas.vector_store"),
                             "M2.4 vector storage has not been implemented")
        self.api = importlib.import_module("northstar_atlas.vector_store")
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="chroma-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
        self.provider = FakeProvider()
        self.config = self.api.IndexConfig(self.provider.model, self.provider.max_input_tokens)
        self.store = self.api.ChromaStore(self.root, self.config)

    def index(self, chunks=None, fingerprint="snapshot-1"):
        return self.store.synchronize(embed_chunks(chunks or [self.hr, self.it], self.provider),
                                      corpus_fingerprint=fingerprint)

    def search(self, **kwargs):
        return self.store.search("access approval", self.provider,
                                 corpus_fingerprint="snapshot-1", reference_date=AS_OF, **kwargs)

    def test_index_counts_repeat_upsert_and_metadata_round_trip(self):
        first = self.index()
        self.assertEqual((first.inserted, first.unchanged, first.deleted, first.total), (2, 0, 0, 2))
        again = self.index()
        self.assertEqual((again.inserted, again.unchanged, again.deleted, again.total), (0, 2, 0, 2))
        hits = self.search(mode="draft_review", top_k=10)
        self.assertEqual({h.chunk.chunk_id for h in hits}, {self.hr.chunk_id, self.it.chunk_id})
        self.assertEqual({h.chunk for h in hits}, {self.hr, self.it})
        for hit in hits:
            self.assertAlmostEqual(hit.cosine_similarity, 1 - hit.cosine_distance)
            self.assertEqual(hit.model, self.provider.model)
        self.assertEqual(self.store.status()["state"], "ready")

    def test_changed_hash_and_removed_document_delete_obsolete_ids(self):
        self.index()
        updated = replace(self.hr, content_hash="a" * 64, chunk_id="new-source-id")
        result = self.index([updated], "snapshot-2")
        self.assertEqual((result.inserted, result.deleted, result.total), (1, 2, 1))
        with self.assertRaisesRegex(self.api.VectorStoreError, "stale"):
            self.search()
        result = self.store.search("access", self.provider, corpus_fingerprint="snapshot-2",
                                   reference_date=AS_OF)
        self.assertEqual(result[0].chunk, updated)

    def test_approved_default_and_effective_date_filter(self):
        future = replace(self.hr, chunk_id="future", effective_date=date(2027, 1, 1))
        self.index([self.hr, self.it, future])
        self.assertEqual([h.chunk.chunk_id for h in self.search(top_k=10)], [self.hr.chunk_id])
        self.assertEqual(len(self.search(mode="draft_review", top_k=10)), 3)
        self.assertEqual(self.store.search("access", self.provider, corpus_fingerprint="snapshot-1",
                                          reference_date=date(2026, 9, 1)), ())

    def test_invalid_queries_modes_limits_and_models(self):
        self.index()
        for query in ("", "   ", None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                self.store.search(query, self.provider, corpus_fingerprint="snapshot-1")
        for args in ({"mode": "all"}, {"top_k": 0}, {"top_k": True}):
            with self.assertRaises(ValueError):
                self.search(**args)
        for model in (replace(self.provider.model, dimension=4),
                      replace(self.provider.model, revision="revision-2")):
            other = FakeProvider()
            other.model = model
            with self.assertRaisesRegex(self.api.VectorStoreError, "model|configuration"):
                self.store.search("query", other, corpus_fingerprint="snapshot-1")

    def test_invalid_vectors_do_not_mutate_ready_index(self):
        self.index()
        for vector in ((1.0, 2.0), (float("nan"), 0.0, 1.0), (1.0, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                self.store.synchronize([EmbeddedChunk(self.hr, vector, self.provider.model)],
                                       corpus_fingerprint="snapshot-2")
        self.assertEqual(self.store.status()["state"], "ready")
        self.assertEqual(len(self.search()), 1)

    def test_failure_after_upsert_marks_incomplete_and_rerun_recovers(self):
        self.index()
        from chromadb.api.models.Collection import Collection
        with patch.object(Collection, "delete", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(self.api.VectorStoreError, "incomplete"):
                self.index([self.hr], "snapshot-2")
        self.assertEqual(self.store.status()["state"], "incomplete")
        with self.assertRaisesRegex(self.api.VectorStoreError, "incomplete"):
            self.search()
        self.index()
        self.assertEqual(self.store.status()["state"], "ready")
        self.assertEqual(len(self.search(mode="draft_review")), 2)

    def test_missing_index_status_does_not_create_database(self):
        self.assertEqual(self.store.status()["state"], "absent")
        self.assertFalse((self.root / ".local/chroma").exists())

    def test_rerun_repairs_malformed_metadata_in_owned_collection(self):
        self.index()
        with self.store._lock():
            collection = self.store._collection()
            collection.add(ids=["unexpected-record"], documents=["invalid local record"],
                           embeddings=[[1.0, 0.0, 0.0]], metadatas=[{"unexpected": "metadata"}])
        self.assertEqual(self.store.status()["state"], "incomplete")
        report = self.index()
        self.assertEqual(report.deleted, 1)
        self.assertEqual(self.store.status()["state"], "ready")

    def test_modified_persisted_vector_is_reported_and_reindex_repairs_it(self):
        self.index()
        with self.store._lock():
            collection = self.store._collection()
            collection.update(ids=[self.hr.chunk_id], embeddings=[[1.0, 0.0, 0.0]])
        self.assertEqual(self.store.status()["state"], "incomplete")
        with self.assertRaisesRegex(self.api.VectorStoreError, "integrity|incomplete"):
            self.search()
        self.index()
        self.assertEqual(self.store.status()["state"], "ready")

    def test_lock_prevents_overlapping_local_operations(self):
        self.index()
        with self.store._lock():
            with self.assertRaisesRegex(self.api.VectorStoreError, "busy"):
                self.search()

    def test_full_sized_float32_batch_can_reindex_and_reopen(self):
        chunks = [replace(self.hr, chunk_id=f"batch-{i}", text=f"{self.hr.text}\nFixture {i}.")
                  for i in range(189)]
        self.assertEqual(self.index(chunks).total, 189)
        self.assertEqual(self.index(chunks).unchanged, 189)
        reopened = self.api.ChromaStore(self.root, self.config)
        self.assertEqual(reopened.status()["state"], "ready")

    def test_configuration_mismatch_rejects_reopening_for_writes(self):
        self.index()
        for config in (replace(self.config, model=replace(self.config.model, revision="other")),
                       replace(self.config, token_limit=128)):
            other = self.api.ChromaStore(self.root, config)
            with self.assertRaisesRegex(self.api.VectorStoreError, "configuration"):
                other.synchronize(embed_chunks([self.hr], self.provider), corpus_fingerprint="snapshot-1")
        self.assertEqual(self.store.status()["state"], "ready")

    def test_reopen_in_separate_process(self):
        self.index()
        code = ('from pathlib import Path; from northstar_atlas.vector_store import ChromaStore, IndexConfig; '
                'from northstar_atlas.embeddings import EmbeddingModel; '
                'from types import SimpleNamespace; from datetime import date; import sys,json; '
                'model=EmbeddingModel("fake-test-only","revision-1",3,"l2"); '
                's=ChromaStore(Path(sys.argv[1]),IndexConfig(model,256)); '
                'provider=SimpleNamespace(model=model,embed_query=lambda text:json.loads(sys.argv[2])); '
                'hits=s.search("known query",provider,corpus_fingerprint="snapshot-1",'
                'reference_date=date(2026,10,22)); '
                'print(s.status()["state"],s.status()["count"]); '
                'print(hits[0].chunk.chunk_id,hits[0].cosine_distance)')
        vector = json.dumps(self.provider.embed_query(self.hr.text))
        result = subprocess.run([sys.executable, "-B", "-c", code, str(self.root), vector],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready 2", result.stdout)
        self.assertIn(self.hr.chunk_id, result.stdout)
        self.assertAlmostEqual(float(result.stdout.splitlines()[-1].split()[-1]), 0.0, places=5)

    def refined_fixture(self):
        provider = FakeProvider()
        provider.max_input_tokens = 36
        text = "## 1. Authorization\n\n" + (
            "Record the authorized approval before executing the change within its permitted window.\n\n" * 5)
        original = replace(self.hr, text=text, source_start=0, source_end=len(text))
        pieces = refine_chunks([original], provider).require_complete()
        self.assertGreater(len(pieces), 1)
        self.config = self.api.IndexConfig(provider.model, 36)
        self.store = self.api.ChromaStore(self.root, self.config)
        self.provider = provider
        return pieces

    def test_sibling_context_preserves_exact_spans_and_has_no_invented_scores(self):
        pieces = self.refined_fixture()
        self.index(pieces)
        hits = self.search(top_k=1)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual({c.chunk_id for c in hit.context}, set(hit.chunk.required_context_ids))
        self.assertEqual({hit.chunk, *hit.context}, set(pieces))
        for chunk in hit.context:
            self.assertFalse(hasattr(chunk, "cosine_distance"))

    def test_missing_and_cross_source_siblings_are_rejected_before_write(self):
        pieces = self.refined_fixture()
        for invalid in (pieces[:-1], (replace(pieces[0], content_hash="b" * 64),) + pieces[1:],
                        (replace(pieces[0], status="Draft — Proposed (Synthetic)", effective_date=None),) + pieces[1:]):
            with self.assertRaisesRegex(self.api.VectorStoreError, "context|sibling"):
                self.index(invalid)
        self.assertEqual(self.store.status()["state"], "absent")

    def test_corrupt_or_missing_stored_records_block_retrieval(self):
        self.index()
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=str(self.root / ".local/chroma"),
                    settings=Settings(anonymized_telemetry=False, _env_file=None,
                        chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
                        chroma_otel_granularity="none", chroma_otel_collection_endpoint="", allow_reset=False))
        collection = client.get_collection(self.api.COLLECTION_NAME, embedding_function=None)
        collection.delete(ids=[self.it.chunk_id])
        with self.assertRaisesRegex(self.api.VectorStoreError, "incomplete|integrity"):
            self.search()
        self.assertEqual(self.store.status()["state"], "incomplete")

    def test_external_symlink_database_is_rejected(self):
        (self.root / ".local").mkdir()
        (self.root / ".local/chroma").symlink_to(ROOT / ".local")
        with self.assertRaisesRegex(self.api.VectorStoreError, "outside|symlink"):
            self.index()

    def test_manifest_and_source_hash_validation_precedes_any_embedding_or_database(self):
        (self.root / "data/policies").mkdir(parents=True)
        shutil.copy(ROOT / "data/corpus_manifest.json", self.root / "data/corpus_manifest.json")
        for source in (ROOT / "data/policies").glob("*.md"):
            shutil.copy(source, self.root / "data/policies" / source.name)
        path = self.root / "data/policies/HR-001.md"
        path.write_text(path.read_text() + "\nTampered.\n")
        with patch.object(self.provider, "embed_documents", side_effect=AssertionError("must not encode")):
            with self.assertRaisesRegex(ValueError, "hash"):
                self.api.index_corpus(self.root, self.provider)
        self.assertFalse((self.root / ".local/chroma").exists())


@unittest.skipUnless(os.environ.get("NORTHSTAR_RUN_MODEL_TESTS") == "1", "cached MiniLM integration is opt-in")
class RealChromaTests(unittest.TestCase):
    def test_cached_minilm_plat_context_and_persisted_cosine_search(self):
        from northstar_atlas.local_embeddings import MiniLMProvider
        from northstar_atlas.vector_store import ChromaStore, IndexConfig
        provider = MiniLMProvider(ROOT)
        documents = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        originals = [c for d in documents for c in chunk_document(d).chunks
                     if d.document_id == "PLAT-001" and c.section_id == "6.1"]
        pieces = refine_chunks(originals, provider).require_complete()
        with tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="chroma-real-") as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
            config = IndexConfig(provider.model, provider.max_input_tokens)
            store = ChromaStore(root, config)
            store.synchronize(embed_chunks(pieces, provider), corpus_fingerprint="verified-test-subset")
            reopened = ChromaStore(root, config)
            hits = reopened.search("Who approves a Terraform production plan?", provider,
                    corpus_fingerprint="verified-test-subset", mode="draft_review", top_k=1, reference_date=AS_OF)
            self.assertEqual(len(hits), 1)
            self.assertEqual({hits[0].chunk, *hits[0].context}, set(pieces))
            text = "\n".join(c.text for c in (hits[0].chunk, *hits[0].context))
            self.assertIn("authorized", text)
            self.assertIn("Change Management", text)
            self.assertIn("scope", text)
            self.assertTrue(math.isfinite(hits[0].cosine_distance))
            self.assertEqual(reopened.search("Terraform", provider,
                    corpus_fingerprint="verified-test-subset", reference_date=AS_OF), ())


if __name__ == "__main__":
    unittest.main()
