"""Lexical scoring and source integrity; no model or provider calls."""
from dataclasses import asdict, replace
from datetime import date
import json
import math
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from test_context import AS_OF, MODEL, ROOT, chunk, family
from northstar_atlas.embeddings import EmbeddedChunk
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.vector_store.records import IndexConfig, VectorStoreError, metadata_for
from northstar_atlas.vector_store.preparation import freeze_snapshot
from northstar_atlas.vector_store.catalog import ChunkCatalog, load_catalog
from northstar_atlas.vector_store.bm25 import BM25Store, tokenize
from northstar_atlas.vector_store.contracts import RankedHit, score_fields

CONFIG = IndexConfig(MODEL, 256)


def catalog(originals, pieces=None):
    return ChunkCatalog(tuple(originals), tuple(pieces or originals), CONFIG, "fixture")


class BM25Tests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        guard.start()
        self.addCleanup(guard.stop)

    def search(self, store, query, **kwargs):
        return store.search(query, None, corpus_fingerprint="fixture", reference_date=AS_OF,
                            **({"mode": "draft_review"} | kwargs))

    def test_unicode_acronyms_numbers_negation_and_punctuation(self):
        self.assertEqual(tokenize("ＡＫＳ IT-001 CI/CD k8s NOT café_temp"),
                         ("aks", "it", "001", "ci", "cd", "k8s", "not", "café", "temp"))

    def test_hand_computed_score_and_unique_query_terms(self):
        a, b = chunk("apple apple", key="a"), chunk("pear pear", key="b", section="4.2")
        store = BM25Store(catalog([a, b]))
        hits = self.search(store, "apple apple")
        self.assertEqual([h.chunk for h in hits], [a])
        self.assertAlmostEqual(hits[0].score, math.log(2) * 1.375)
        self.assertEqual(hits, self.search(store, "apple"))
        self.assertEqual(hits[0].score_kind, "bm25")
        self.assertNotIn("cosine_distance", score_fields(hits[0]))

    def test_term_saturation_and_length_normalization(self):
        a = chunk("apple", key="a")
        b = chunk("apple filler filler filler", key="b", section="4.2")
        store = BM25Store(catalog([a, b]))
        hits = self.search(store, "apple")
        self.assertEqual([h.chunk.chunk_id for h in hits], ["a", "b"])
        no_norm = BM25Store(catalog([a, b]), b=0)
        self.assertEqual(self.search(no_norm, "apple")[0].score,
                         self.search(no_norm, "apple")[1].score)

    def test_deterministic_tie_order_ignores_input_order(self):
        a = chunk("same text", key="a")
        b = chunk("same text", key="b", section="4.2")
        forward = self.search(BM25Store(catalog([a, b])), "same")
        reverse = self.search(BM25Store(catalog([b, a])), "same")
        self.assertEqual(forward, reverse)
        self.assertEqual([h.chunk.chunk_id for h in forward], ["a", "b"])

    def test_approved_statistics_exclude_drafts_and_future_policies(self):
        hr = chunk("access approval", key="hr", doc="HR-001", status=APPROVED, effective=AS_OF)
        draft = chunk("access access access", key="draft")
        future = chunk("access", key="future", doc="SEC-001", status=APPROVED, effective=date(2099,1,1))
        all_store = BM25Store(catalog([hr, draft, future]))
        approved = self.search(all_store, "access", mode="approved_only")
        self.assertEqual(approved, self.search(BM25Store(catalog([hr])), "access", mode="approved_only"))
        self.assertEqual([h.chunk for h in approved], [hr])
        self.assertEqual(len(self.search(all_store, "access")), 3)
        self.assertEqual(all_store.search("access", None, corpus_fingerprint="fixture", mode="approved_only",
                                         reference_date=date(2026,10,21)), ())

    def test_no_overlap_and_no_lexical_tokens_return_no_zero_fillers(self):
        store = BM25Store(catalog([chunk("a real rule")]))
        self.assertEqual(self.search(store, "zzzz"), ())
        self.assertEqual(self.search(store, "?!"), ())
        self.assertEqual(self.search(store, "rule", mode="approved_only"), ())

    def test_invalid_input_and_configuration(self):
        c = catalog([chunk()])
        for kwargs in ({"k1": 0}, {"k1": float("nan")}, {"b": 1.1}, {"b": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(VectorStoreError): BM25Store(c, **kwargs)
        store = BM25Store(c)
        for q in ("", "  ", None):
            with self.subTest(q=q), self.assertRaises(VectorStoreError): self.search(store, q)
        for kwargs in ({"top_k": True}, {"top_k": 0}, {"mode": "all"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(VectorStoreError): self.search(store, "rule", **kwargs)
        with self.assertRaises(VectorStoreError):
            store.search("rule", None, corpus_fingerprint="stale", reference_date=AS_OF)
        with self.assertRaises(VectorStoreError):
            BM25Store(catalog([chunk("!!!")]))

    def test_metadata_citations_and_siblings_are_unchanged(self):
        original = chunk("All conditions.\nTechnical approval.\nStop outside scope.\n")
        pieces = family(original, [16, 36])
        store = BM25Store(catalog([original], pieces))
        hits = self.search(store, "technical", top_k=1)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(set(c.chunk_id for c in (h.chunk, *h.context)), set(c.chunk_id for c in pieces))
        self.assertEqual(h.chunk.citation, original.citation)
        self.assertEqual(h.model, MODEL)
        self.assertEqual(tuple(store.catalog.chunks), tuple(pieces))
        with self.assertRaises(TypeError): store.catalog.by_id["injected"] = original

    def test_catalog_rejects_tampered_metadata_spans_and_missing_family(self):
        original = chunk("Intro.\nApproval.\nStop.\n")
        pieces = family(original, [7])
        bad_sets = [pieces[:1], [pieces[0], pieces[0]],
                    [replace(pieces[0], title="Invented"), pieces[1]],
                    [pieces[0], replace(pieces[1], text=pieces[1].text.replace("Intro", "Wrong"))],
                    [replace(pieces[0], source_end=999), pieces[1]],
                    [replace(pieces[0], content_hash="b"*64), pieces[1]]]
        for bad in bad_sets:
            with self.subTest(bad=bad), self.assertRaises(VectorStoreError): catalog([original], bad)
        with self.assertRaises(VectorStoreError): catalog([original], [chunk(key="unknown")])
        with self.assertRaises(VectorStoreError): catalog([original, original], pieces)

    def test_neutral_scores_are_validated_without_fake_cosine(self):
        c = chunk()
        for kind, value in (("rrf", float("inf")), ("bm25", -1), ("unknown", 1), ("rrf", True)):
            with self.subTest(kind=kind), self.assertRaises(VectorStoreError):
                score_fields(RankedHit(c, value, kind, MODEL, ()))

    def test_snapshot_reader_preserves_digest_and_rejects_escape(self):
        from northstar_atlas.vector_store.preparation import load_frozen_snapshot
        snapshot = freeze_snapshot([EmbeddedChunk(chunk(), (1.,0.,0.), MODEL)], CONFIG, "fixture")
        data = {"snapshot_id": snapshot.snapshot_id, "corpus_fingerprint": "fixture",
                "config": asdict(CONFIG), "records": [
                    {"id": r.chunk_id, "text": r.chunk.text, "metadata": metadata_for(r.chunk, r.model),
                     "vector": list(r.vector)} for r in snapshot.records]}
        with tempfile.TemporaryDirectory(dir=ROOT/".local/tmp") as temp:
            root = Path(temp)
            folder = root/".local/pinecone/snapshots"; folder.mkdir(parents=True)
            path = folder/f"{snapshot.snapshot_id}.json"; path.write_text(json.dumps(data))
            self.assertEqual(load_frozen_snapshot(root, snapshot.namespace), snapshot)
            data["records"][0]["text"] = "Changed"
            path.write_text(json.dumps(data))
            with self.assertRaises(VectorStoreError): load_frozen_snapshot(root, snapshot.namespace)
            with self.assertRaises(VectorStoreError): load_frozen_snapshot(root, "../escape")


if __name__ == "__main__": unittest.main()
