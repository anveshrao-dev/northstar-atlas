"""Rank fusion and existing context/QA compatibility with deterministic fixtures."""
from contextlib import contextmanager
from dataclasses import replace
import socket
import unittest
from unittest.mock import patch

from test_context import AS_OF, MODEL, ROOT, CharacterCounter, chunk, family, hit
from test_bm25 import catalog
from test_qa import FakeLLM, authored_claims
from northstar_atlas.context import build_context
from northstar_atlas.generation import prepare_answer
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.vector_store.bm25 import BM25Store
from northstar_atlas.vector_store.contracts import RankedHit, SearchHit, score_fields
from northstar_atlas.vector_store.hybrid import HybridStore, fuse_results
from northstar_atlas.vector_store.records import VectorStoreError


def lexical(c, score=1, siblings=()): return RankedHit(c, score, "bm25", MODEL, tuple(siblings))


class DenseFixture:
    def __init__(self, hits=(), error=None): self.hits, self.error, self.calls = tuple(hits), error, []
    def search(self, question, provider, **kwargs):
        self.calls.append((question, provider, kwargs))
        if self.error: raise self.error
        return self.hits


class HybridTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        guard.start(); self.addCleanup(guard.stop)

    def fuse(self, dense, lexical_hits, c):
        return fuse_results(dense, lexical_hits, c, mode="draft_review", reference_date=AS_OF)

    def test_hand_computed_rrf_and_ties(self):
        a, b, c = [chunk(key=k, section=f"4.{i}") for i,k in enumerate("abc",1)]
        result = self.fuse([hit(a), hit(b)], [lexical(b), lexical(a), lexical(c)], catalog([a,b,c]))
        self.assertEqual([h.chunk.chunk_id for h in result], ["a","b","c"])
        self.assertAlmostEqual(result[0].score, 1/61+1/62)
        self.assertAlmostEqual(result[2].score, 1/63)
        self.assertEqual([(v.channel,v.rank) for v in result[0].contributions], [("dense",1),("bm25",2)])
        self.assertNotIn("cosine_similarity", score_fields(result[0]))

    def test_cross_channel_ids_merge_but_duplicates_within_channel_fail(self):
        a = chunk(); c = catalog([a])
        self.assertEqual(len(self.fuse([hit(a)], [lexical(a)], c)),1)
        for dense, lex in (([hit(a),hit(a)],[]),([],[lexical(a),lexical(a)])):
            with self.assertRaises(VectorStoreError): self.fuse(dense,lex,c)
        with self.assertRaises(VectorStoreError): self.fuse([hit(a)],[lexical(replace(a,text="forged"))],c)

    def test_wrong_channel_score_unknown_id_and_ineligible_hits_rejected(self):
        a=chunk(); c=catalog([a])
        for dense, lex in (([lexical(a)],[]),([],[hit(a)]),([hit(chunk(key="unknown"))],[])):
            with self.assertRaises(VectorStoreError): self.fuse(dense,lex,c)
        with self.assertRaises(VectorStoreError):
            fuse_results([hit(a)], [], c, mode="approved_only", reference_date=AS_OF)

    def test_siblings_never_add_votes_or_replace_missing_context(self):
        original=chunk("Intro.\nApproval.\nStop.\n"); parts=family(original,[7])
        c=catalog([original],parts)
        result=self.fuse([hit(parts[0],parts[1:])],[],c)
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(result[0].score,1/61)
        self.assertEqual(result[0].context,tuple(parts[1:]))
        for bad in (hit(parts[0]),hit(parts[0],[replace(parts[1],status=APPROVED,effective_date=AS_OF)])):
            with self.assertRaises(VectorStoreError): self.fuse([bad],[],c)

    def test_hybrid_requests_twenty_and_uses_one_question_mode_date(self):
        a=chunk("approval")
        dense=DenseFixture([hit(a)])
        store=HybridStore(dense,BM25Store(catalog([a])))
        provider=object()
        result=store.search("approval",provider,corpus_fingerprint="fixture",mode="draft_review",reference_date=AS_OF)
        self.assertEqual(len(result),1)
        self.assertEqual(dense.calls,[("approval",provider,{"corpus_fingerprint":"fixture","mode":"draft_review",
                                                        "top_k":20,"reference_date":AS_OF})])

    def test_channel_errors_stop_no_fallback_and_empty_lexical_is_valid(self):
        c=catalog([chunk("approval")]); lexical_store=BM25Store(c)
        with self.assertRaises(RuntimeError):
            HybridStore(DenseFixture(error=RuntimeError("failed")),lexical_store).search(
                "approval",object(),corpus_fingerprint="fixture",mode="draft_review",reference_date=AS_OF)
        result=HybridStore(DenseFixture([hit(c.chunks[0])]),lexical_store).search(
            "zzzz",object(),corpus_fingerprint="fixture",mode="draft_review",reference_date=AS_OF)
        self.assertEqual(len(result[0].contributions),1)

    def test_neutral_context_identical_to_dense_and_budget_guard_preserved(self):
        a=chunk(); kwargs=dict(counter=CharacterCounter(),max_tokens=10000,mode="draft_review",reference_date=AS_OF)
        self.assertEqual(build_context([hit(a)],**kwargs),build_context([lexical(a)],**kwargs))
        p=build_context([lexical(a)],**(kwargs|{"max_tokens":build_context([],**kwargs).token_count}))
        self.assertFalse(p.groups)
        self.assertEqual(p.warnings[0].code,"budget_excluded")

    def test_dense_legacy_score_output_unchanged(self):
        self.assertEqual(score_fields(hit(chunk())),{"cosine_distance":.2,"cosine_similarity":.8})
        for bad in (float("nan"),float("inf")):
            with self.assertRaises(VectorStoreError): score_fields(replace(hit(chunk()),cosine_distance=bad))

    def test_qa_accepts_lexical_evidence_and_withholds_invalid_output(self):
        from northstar_atlas.qa import QAService
        a=chunk("Only within approved scope.")
        store=BM25Store(catalog([a]))
        hits=store.search("scope",None,corpus_fingerprint="fixture",mode="draft_review",reference_date=AS_OF)
        prepared=prepare_answer("scope",hits,mode="draft_review",reference_date=AS_OF)
        provider=FakeLLM(authored_claims(prepared))
        service=QAService(store,None,provider,corpus_fingerprint="fixture",strategy="bm25")
        result=service.ask("scope",mode="draft_review",reference_date=AS_OF)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.status,"not_assessed")
        self.assertEqual(result.diagnostics.retrieval_strategy,"bm25")
        self.assertEqual(result.public.citations[0].content_hash,a.content_hash)
        provider.body="SECRET_REJECTED malformed"
        rejected=service.ask("scope",mode="draft_review",reference_date=AS_OF)
        self.assertFalse(rejected.accepted)
        self.assertNotIn("SECRET_REJECTED",str(rejected))

    def test_opt_in_factory_bm25_avoids_models_and_dense_stores(self):
        from northstar_atlas.qa import open_retriever
        c=catalog([chunk()])
        with patch("northstar_atlas.vector_store.catalog.load_catalog",return_value=c), \
             patch("northstar_atlas.local_embeddings.MiniLMProvider",side_effect=AssertionError("model forbidden")):
            with open_retriever(ROOT,"chroma",strategy="bm25",snapshot_id="fixture") as (store,embedder,fp):
                self.assertIsInstance(store,BM25Store)
                self.assertIsNone(embedder)
                self.assertEqual(fp,"fixture")


if __name__ == "__main__": unittest.main()
