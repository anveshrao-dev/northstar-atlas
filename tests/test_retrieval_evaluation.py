"""Pure metric fixtures plus pinned-dataset and output-isolation checks."""

from dataclasses import replace
from datetime import date
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, load_documents
from northstar_atlas.ingestion.refiner import refine_chunks
from northstar_atlas.vector_store import SearchHit


ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 22)
MODEL = EmbeddingModel("test-evaluation", "v1", 3, "l2")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = chunk_document(load_documents(ROOT, reference_date=AS_OF)[0]).chunks[0]

    def setUp(self):
        package = importlib.util.find_spec("northstar_atlas.evaluation")
        self.assertIsNotNone(package, "M2.5 retrieval evaluation is not implemented")
        self.api = importlib.import_module("northstar_atlas.evaluation.retrieval")

    def chunk(self, section, *, doc="HR-001", start=0, text="Complete rule.\n", **kwargs):
        return replace(self.base, document_id=doc, section_id=section,
                       chunk_id=f"{doc}:{section}:{start}", text=text,
                       source_start=start, source_end=start + len(text), **kwargs)

    def question(self, refs, *, mode="approved_only", category="A", behavior="answer"):
        return {"question_id": "EVAL-001", "category": category, "question": "Test question?",
                "retrieval_mode": mode, "expected_behavior": behavior,
                "required_evidence": [{"document_id": d, "section_id": s} for d, s in refs]}

    def hit(self, chunk, *context, distance=0.2):
        return SearchHit(chunk, distance, MODEL, tuple(context))

    def score(self, q, hits, canonical, prepared=None):
        catalog = self.api.EvidenceCatalog(canonical, prepared)
        return self.api.score_question(q, hits, catalog, reference_date=AS_OF, latency_seconds=0.125)

    def test_known_metrics_and_first_complete_evidence_rank(self):
        one, two, other = [self.chunk(s, start=i * 100) for i, s in enumerate(["1", "2", "3"])]
        result = self.score(self.question([("HR-001", "1"), ("HR-001", "2")]),
                            [self.hit(other), self.hit(one)], [one, two, other])
        self.assertEqual(result["direct"]["recall_at_5"], 0.5)
        self.assertEqual(result["direct"]["hit_at_5"], 1)
        self.assertEqual(result["direct"]["full_evidence_coverage_at_5"], 0)
        self.assertEqual(result["direct"]["reciprocal_rank_at_5"], 0.5)
        self.assertEqual(result["direct"]["missing_evidence"], [{"document_id": "HR-001", "section_id": "2"}])
        self.assertEqual(result["latency_seconds"], 0.125)

    def test_duplicates_consume_slots_without_double_credit(self):
        one, two = self.chunk("1"), self.chunk("2", start=100)
        result = self.score(self.question([("HR-001", "1"), ("HR-001", "2")]),
                            [self.hit(one)] * 5 + [self.hit(two)], [one, two])
        self.assertEqual(len(result["retrieved"]), 5)
        self.assertEqual(result["direct"]["recall_at_5"], 0.5)
        self.assertEqual(result["expanded"]["recall_at_5"], 0.5)

    def test_parent_requires_all_descendants_and_never_matches_another_document(self):
        a, b = self.chunk("4.1"), self.chunk("4.2", start=100)
        misleading = self.chunk("40.1", start=200)
        other_doc = self.chunk("4.2", doc="IT-001", start=100)
        catalog = [a, b, misleading, other_doc]
        q = self.question([("HR-001", "4")])
        self.assertEqual(self.score(q, [self.hit(a)], catalog)["direct"]["recall_at_5"], 0)
        self.assertEqual(self.score(q, [self.hit(a), self.hit(other_doc), self.hit(misleading)], catalog)["direct"]["recall_at_5"], 0)
        good = self.score(q, [self.hit(a), self.hit(b)], catalog)
        self.assertEqual(good["direct"]["recall_at_5"], 1)
        self.assertEqual(good["direct"]["reciprocal_rank_at_5"], 0.5)

    def test_parent_heading_does_not_cover_child_body(self):
        parent = self.chunk("4", text="## 4. Parent\n")
        child = self.chunk("4.1", start=len(parent.text), text="Full approval conditions.\n")
        result = self.score(self.question([("HR-001", "4")]), [self.hit(parent)], [parent, child])
        self.assertEqual(result["direct"]["recall_at_5"], 0)
        self.assertEqual(result["direct"]["section_id_recall_at_5"], 1)

    def fragments(self):
        class Counter:
            model = MODEL
            max_input_tokens = 26
            def token_counts(self, texts):
                return [len(t.split()) + 2 for t in texts]
        text = "## 1. Conditions\n\n" + (
            "Only approved operators may execute within the authorized scope and window.\n\n" * 4)
        original = self.chunk("1", text=text)
        pieces = refine_chunks([original], Counter()).require_complete()
        self.assertGreater(len(pieces), 1)
        return original, pieces

    def test_partial_section_and_sibling_expansion_accounting(self):
        original, pieces = self.fragments()
        q = self.question([("HR-001", "1")])
        result = self.score(q, [self.hit(pieces[0], *pieces[1:])], [original], pieces)
        self.assertEqual(result["direct"]["section_id_recall_at_5"], 1)
        self.assertEqual(result["direct"]["recall_at_5"], 0)
        self.assertEqual(result["expanded"]["recall_at_5"], 1)
        self.assertEqual(result["expanded"]["reciprocal_rank_at_5"], 1)
        self.assertEqual(len(result["retrieved"]), 1)
        for context in result["sibling_context"]:
            self.assertIsNone(context["rank"])
            self.assertIsNone(context["cosine_distance"])
            self.assertEqual(context["trigger_ranks"], [1])

    def test_expanded_rr_uses_trigger_prefix_not_a_fabricated_sibling_rank(self):
        original, pieces = self.fragments()
        other = self.chunk("2", start=1000)
        result = self.score(self.question([("HR-001", "1")]),
                            [self.hit(other), self.hit(pieces[0], *pieces[1:])],
                            [original, other], [*pieces, other])
        self.assertEqual(result["expanded"]["reciprocal_rank_at_5"], 0.5)
        self.assertEqual(result["direct"]["reciprocal_rank_at_5"], 0)

    def test_sibling_audit_preserves_invalid_and_valid_occurrences(self):
        original, pieces = self.fragments()
        other = self.chunk("2", start=1000)
        result = self.score(self.question([("HR-001", "1")]),
                            [self.hit(other, pieces[1]), self.hit(pieces[0], *pieces[1:])],
                            [original, other], [*pieces, other])
        sibling = next(c for c in result["sibling_context"] if c["chunk_id"] == pieces[1].chunk_id)
        self.assertEqual(result["expanded"]["recall_at_5"], 1)
        self.assertTrue(sibling["eligible_verified"])
        self.assertEqual(sibling["trigger_ranks"], [1, 2])
        self.assertEqual(sibling["valid_trigger_ranks"], [2])
        self.assertEqual([o["eligible_verified"] for o in sibling["occurrences"]], [False, True])
        self.assertTrue(any(v["reason"] == "unexpected_context" for v in result["violations"]))

    def test_all_direct_fragments_jointly_cover_section(self):
        original, pieces = self.fragments()
        result = self.score(self.question([("HR-001", "1")]),
                            [self.hit(p, *(c for c in pieces if c != p)) for p in pieces], [original], pieces)
        self.assertEqual(result["direct"]["recall_at_5"], 1)
        self.assertEqual(result["direct"]["reciprocal_rank_at_5"], 1 / len(pieces))

    def test_authority_leakage_excluded_from_both_views(self):
        draft = self.chunk("1", status=DRAFT, effective_date=None)
        future = self.chunk("2", start=100, effective_date=date(2027, 1, 1))
        good = self.chunk("3", start=200)
        q = self.question([("HR-001", "1"), ("HR-001", "2")])
        result = self.score(q, [self.hit(draft), self.hit(good, future)], [draft, future, good])
        self.assertEqual(result["direct"]["recall_at_5"], 0)
        self.assertEqual(result["expanded"]["recall_at_5"], 0)
        self.assertGreaterEqual(len(result["violations"]), 2)

    def test_draft_review_keeps_status_and_clarify_abstain_purpose(self):
        draft = self.chunk("1", status=DRAFT, effective_date=None)
        for category, behavior, purpose in [("C", "clarify", "conditional_rules"), ("D", "abstain", "corpus_limits")]:
            q = self.question([("HR-001", "1")], mode="draft_review", category=category, behavior=behavior)
            result = self.score(q, [self.hit(draft)], [draft])
            self.assertEqual(result["direct"]["recall_at_5"], 1)
            self.assertEqual(result["retrieved"][0]["status"], DRAFT)
            self.assertEqual(result["evidence_purpose"], purpose)
            self.assertIsNone(result["answer_behavior_evaluated"])

    def test_unverified_text_or_source_hash_never_earns_credit(self):
        chunk = self.chunk("1")
        for bad in (replace(chunk, content_hash="0" * 64), replace(chunk, text="Changed")):
            result = self.score(self.question([("HR-001", "1")]), [self.hit(bad)], [chunk])
            self.assertEqual(result["direct"]["recall_at_5"], 0)
            self.assertTrue(result["violations"])

    def test_empty_hits_zero_metrics_and_all_missing(self):
        chunk = self.chunk("1")
        result = self.score(self.question([("HR-001", "1")]), [], [chunk])
        for name in ("recall_at_5", "hit_at_5", "full_evidence_coverage_at_5", "reciprocal_rank_at_5"):
            self.assertEqual(result["direct"][name], 0)
        self.assertEqual(result["direct"]["missing_evidence"], result["required_evidence"])

    def test_invalid_questions_references_and_latency_rejected(self):
        chunk = self.chunk("1")
        valid = self.question([("HR-001", "1")])
        for change in ({"required_evidence": []}, {"required_evidence": valid["required_evidence"] * 2},
                       {"required_evidence": [{"document_id": "HR-001", "section_id": "99"}]},
                       {"retrieval_mode": "all"}, {"expected_behavior": "invent"},
                       {"category": "D"}, {"acceptable_alternative_evidence": [1]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score({**valid, **change}, [], [chunk])
        with self.assertRaises(ValueError):
            self.api.score_question(valid, [], self.api.EvidenceCatalog([chunk]),
                                    reference_date=AS_OF, latency_seconds=float("nan"))

    def test_pinned_dataset_validation_and_deterministic_format(self):
        dataset, dataset_hash = self.api.load_dataset(ROOT)
        self.assertEqual(len(dataset["questions"]), 15)
        self.assertEqual(len(dataset_hash), 64)
        self.assertEqual(self.api.format_results({"b": 2, "a": [1]}),
                         self.api.format_results({"a": [1], "b": 2}))
        with self.assertRaises(ValueError):
            self.api.format_results({"bad": float("inf")})

    def test_summary_uses_macro_means_and_separates_behavior_groups(self):
        chunk = self.chunk("1")
        a = self.score(self.question([("HR-001", "1")]), [self.hit(chunk)], [chunk])
        d = self.score(self.question([("HR-001", "1")], mode="draft_review", category="D", behavior="abstain"), [], [chunk])
        summary = self.api.summarize([a, d])
        self.assertEqual(summary["overall"]["direct"]["recall_at_5"], 0.5)
        self.assertEqual(summary["by_category"]["A"]["direct"]["recall_at_5"], 1)
        self.assertEqual(summary["answerable"]["direct"]["recall_at_5"], 1)
        self.assertEqual(summary["by_category"]["D"]["direct"]["recall_at_5"], 0)

    def test_incomplete_or_stale_index_rejected_before_model_loading(self):
        for state in ({"state": "incomplete"}, {"state": "ready", "corpus_fingerprint": "stale"}):
            with self.subTest(state=state), \
                    patch.object(self.api, "load_dataset", return_value=({}, "hash")), \
                    patch.object(self.api, "corpus_snapshot", return_value=([], "current")), \
                    patch.object(self.api, "ChromaStore") as store, \
                    patch.object(self.api, "MiniLMProvider") as provider:
                store.return_value.status.return_value = state
                with self.assertRaisesRegex(ValueError, "will not rebuild"):
                    self.api.evaluate(ROOT, reference_date=AS_OF)
                provider.assert_not_called()
                store.return_value.synchronize.assert_not_called()

    def test_changed_index_or_corpus_discards_evaluation(self):
        chunk = self.chunk("1")
        question = self.question([("HR-001", "1")])
        initial = {"state": "ready", "corpus_fingerprint": "current", "count": 1}
        for changed in ("index", "corpus"):
            store = MagicMock()
            store._lock = MagicMock()
            store.status.side_effect = [initial, {**initial, "count": 2} if changed == "index" else initial]
            store._collection.return_value.configuration_json = {"hnsw": {"space": "cosine"}}
            snapshots = [([object()], "current"), ([object()], "changed" if changed == "corpus" else "current")]
            with self.subTest(changed=changed), \
                    patch.object(self.api, "load_dataset", return_value=({"questions": [question]}, "hash")), \
                    patch.object(self.api, "corpus_snapshot", side_effect=snapshots), \
                    patch.object(self.api, "ChromaStore", return_value=store), \
                    patch.object(self.api, "MiniLMProvider") as provider, \
                    patch.object(self.api, "chunk_document") as canonical, \
                    patch.object(self.api, "refine_chunks") as refined, \
                    patch.object(self.api, "search_corpus", return_value=[self.hit(chunk)]):
                canonical.return_value.chunks = [chunk]
                refined.return_value.require_complete.return_value = [chunk]
                provider.return_value.token_counts.return_value = [1]
                with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                    self.api.evaluate(ROOT, reference_date=AS_OF)
                store.synchronize.assert_not_called()

    def test_artifact_creation_refuses_overwrite_or_outside_symlinks(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="eval-test-") as temp:
            root = Path(temp)
            path = self.api.write_results(root, {"a": 1}, run_name="test")
            self.assertTrue(path.is_relative_to(root / ".local/evaluations/m2.5"))
            with self.assertRaises(FileExistsError):
                self.api.write_results(root, {"a": 2}, run_name="test")
            for name in ("../outside", "/absolute", "has/slash"):
                with self.assertRaises(ValueError):
                    self.api.write_results(root, {}, run_name=name)
        with tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp", prefix="eval-test-") as temp:
            root = Path(temp)
            (root / ".local").symlink_to(ROOT / ".local")
            with self.assertRaises(ValueError):
                self.api.write_results(root, {}, run_name="bad")


if __name__ == "__main__":
    unittest.main()
