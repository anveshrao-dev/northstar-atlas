"""Source coverage and semantic-context contracts are deliberately separate."""

from dataclasses import fields, replace
from datetime import date
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unittest

from northstar_atlas.embeddings import EmbeddingModel, embed_chunks, embed_query
from northstar_atlas.ingestion.chunker import PolicyChunk, chunk_document
from northstar_atlas.ingestion.document_loader import load_documents


ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 22)


class WordCounter:
    """A deterministic budget for structural tests, not a model tokenizer."""

    model = EmbeddingModel("test-counter", "revision-1", 3)
    max_input_tokens = 32

    def token_counts(self, texts):
        return [len(text.split()) + 2 for text in texts]


class RefinerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)[0]

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("northstar_atlas.ingestion.refiner"),
                             "The separate token-aware refinement stage is not implemented")
        self.refiner = importlib.import_module("northstar_atlas.ingestion.refiner")
        self.counter = WordCounter()

    def fixture(self, body):
        header = self.document.content.split("## 1.", 1)[0]
        text = header + "## 1. Conditions\n\n" + body
        document = replace(self.document, content=text,
                           content_hash=hashlib.sha256(text.encode()).hexdigest())
        return chunk_document(document).chunks[0]

    def assert_coverage(self, original, pieces):
        self.assertEqual(self.refiner.reconstruct(original, pieces), original.text)
        spans = []
        for piece in pieces:
            self.assertLessEqual(self.counter.token_counts([piece.text])[0], self.counter.max_input_tokens)
            for segment in piece.segments:
                rendered = piece.text[segment.text_start:segment.text_end]
                self.assertEqual(rendered, original.text[segment.source_start-original.source_start:
                                                         segment.source_end-original.source_start])
                if segment.role == "source":
                    spans.append((segment.source_start, segment.source_end, rendered))
        self.assertEqual("".join(text for _, _, text in sorted(spans)), original.text)

    def test_under_budget_passes_through_without_new_id_or_text(self):
        chunk = self.fixture("Only approved access is permitted.\n")
        result = self.refiner.refine_chunks([chunk], self.counter)
        self.assertIs(result.chunks[0], chunk)
        self.assertFalse(result.issues)
        self.assertEqual(result.require_complete(), result.chunks)

    def test_paragraphs_exact_coverage_metadata_and_required_siblings(self):
        original = self.fixture("Approved operators may perform this action within the recorded window.\n\n"
                                "A window extension requires a new recorded decision by an authorized owner.\n\n"
                                "An incident does not waive these conditions or confer additional authority.\n")
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertFalse(result.issues)
        self.assertGreater(len(result.chunks), 1)
        self.assert_coverage(original, result.chunks)
        for piece in result.chunks:
            self.assertEqual(piece.original_chunk_id, original.chunk_id)
            self.assertEqual(set(piece.required_context_ids), {p.chunk_id for p in result.chunks} - {piece.chunk_id})
            self.assertEqual(piece.citation, original.citation)
            self.assertEqual(piece.content_hash, original.content_hash)
        self.assertTrue(any(s.role == "context" for c in result.chunks for s in c.segments))

    def test_tables_repeat_headers_and_keep_complete_rows_and_qualification(self):
        original = self.fixture("Only within approved scope.\n\n| Actor | Boundary |\n| --- | --- |\n"
                                "| Platform | Technical review does not grant deployment authorization |\n"
                                "| Security | Access review does not grant deployment authorization |\n"
                                "| SRE | Incident coordination does not grant deployment authorization |\n")
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertFalse(result.issues)
        self.assert_coverage(original, result.chunks)
        for piece in result.chunks:
            if "| Platform |" in piece.text or "| Security |" in piece.text or "| SRE |" in piece.text:
                self.assertIn("| Actor | Boundary |\n| --- | --- |", piece.text)
                self.assertIn("Only within approved scope.", piece.text)

    def test_list_items_keep_wrapped_conditions_and_intro(self):
        original = self.fixture("All conditions apply:\n\n"
                                "1. Obtain recorded approval before execution\n   unless an authorized exception is recorded.\n"
                                "2. Verify the window before execution\n   and stop if that approved window has expired.\n"
                                "3. Record evidence after execution\n   without placing credentials in ordinary logs.\n")
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertFalse(result.issues)
        self.assert_coverage(original, result.chunks)
        for piece in result.chunks:
            self.assertIn("All conditions apply:", piece.text)
            if "1. Obtain" in piece.text:
                self.assertIn("unless an authorized exception is recorded.", piece.text)

    def test_long_paragraph_uses_sentences_without_mid_sentence_cuts(self):
        sentence = "Only approved operators may apply changes within the recorded scope and window. "
        original = self.fixture(sentence * 5 + "\n")
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertFalse(result.issues)
        self.assert_coverage(original, result.chunks)
        for piece in result.chunks:
            for segment in piece.segments:
                if segment.role == "source":
                    self.assertTrue(piece.text[segment.text_start:segment.text_end].rstrip().endswith("."))

    def test_unsplittable_sentence_table_row_and_fence_are_explicit_failures(self):
        for body in ["condition " * 80, "| Role | Rule |\n| --- | --- |\n| Owner | " + "condition " * 80 + "|\n",
                     "```text\n" + "condition " * 80 + "\n```\n"]:
            with self.subTest(body=body[:20]):
                original = self.fixture(body)
                result = self.refiner.refine_chunks([original], self.counter)
                self.assertEqual(result.chunks, ())
                self.assertEqual(result.issues[0].original_chunk_id, original.chunk_id)
                self.assertIn("unsafe", result.issues[0].reason.lower())
                with self.assertRaises(self.refiner.RefinementError):
                    result.require_complete()

    def test_ids_deterministic_and_sensitive_to_source_and_model(self):
        original = self.fixture(("A complete rule requires recorded authorization.\n\n") * 8)
        first = self.refiner.refine_chunks([original], self.counter)
        second = self.refiner.refine_chunks([original], self.counter)
        self.assertEqual(first, second)
        ids = {c.chunk_id for c in first.chunks}
        self.assertEqual(len(ids), len(first.chunks))
        self.assertNotIn(original.chunk_id, ids)
        other = replace(original, content_hash="0" * 64)
        self.assertFalse(ids & {c.chunk_id for c in self.refiner.refine_chunks([other], self.counter).chunks})
        self.counter.model = replace(self.counter.model, revision="revision-2")
        self.assertFalse(ids & {c.chunk_id for c in self.refiner.refine_chunks([original], self.counter).chunks})

    def test_reconstruction_rejects_missing_duplicate_or_tampered_pieces(self):
        original = self.fixture(("Record approval before execution and retain the evidence.\n\n") * 7)
        pieces = self.refiner.refine_chunks([original], self.counter).require_complete()
        for invalid in [pieces[:-1], pieces + (pieces[0],),
                        (replace(pieces[0], text="CORRUPTED" + pieces[0].text),) + pieces[1:]]:
            with self.assertRaises(self.refiner.RefinementError):
                self.refiner.reconstruct(original, invalid)

    def test_duplicate_inputs_and_invalid_budgets_are_rejected(self):
        original = self.fixture("A rule.\n")
        with self.assertRaises(self.refiner.RefinementError):
            self.refiner.refine_chunks([original, original], self.counter)
        self.counter.max_input_tokens = 0
        with self.assertRaises(self.refiner.RefinementError):
            self.refiner.refine_chunks([original], self.counter)

    def test_empty_table_reports_issue_instead_of_crashing(self):
        original = self.fixture("| Actor | Boundary |\n| --- | --- |\n\n" +
                                "Record the approved decision before executing the operation.\n\n" * 5)
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertEqual(result.chunks, ())
        self.assertIn("table", result.issues[0].reason)

    def test_indented_list_item_is_not_misclassified_as_splittable_prose(self):
        self.counter.max_input_tokens = 24
        original = self.fixture("  - Operators may proceed after obtaining approval from the accountable service owner. "
                                "They must stop whenever approval expires or an incident changes the approved risk.\n")
        result = self.refiner.refine_chunks([original], self.counter)
        self.assertEqual(result.chunks, ())
        self.assertIn("list item", result.issues[0].reason)

    def test_abbreviation_and_initial_do_not_create_mid_sentence_boundaries(self):
        self.counter.max_input_tokens = 24
        for name in ("Mr. Smith", "Dr. Smith", "J. Smith", "e.g. Smith"):
            with self.subTest(name=name):
                original = self.fixture("Only the authorized manager may approve production changes after reviewing "
                                        f"the written evidence supplied by {name} and the affected service owner.\n")
                result = self.refiner.refine_chunks([original], self.counter)
                self.assertEqual(result.chunks, ())
                self.assertIn("sentence", result.issues[0].reason)


@unittest.skipUnless(os.environ.get("NORTHSTAR_RUN_MODEL_TESTS") == "1", "Enable cached-model checks explicitly")
class RealRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from northstar_atlas.local_embeddings import MiniLMProvider
        cls.provider = MiniLMProvider(ROOT)
        cls.documents = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        cls.originals = tuple(c for d in cls.documents for c in chunk_document(d).chunks)

    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("northstar_atlas.ingestion.refiner"))
        self.refiner = importlib.import_module("northstar_atlas.ingestion.refiner")
        self.result = self.refiner.refine_chunks(self.originals, self.provider)
        self.chunks = self.result.require_complete()

    def test_entire_corpus_fits_actual_tokenizer_and_reconstructs_exactly(self):
        self.assertEqual(len(self.originals), 151)
        self.assertEqual(sum(n > 256 for n in self.provider.token_counts([c.text for c in self.originals])), 25)
        self.assertLessEqual(max(self.provider.token_counts([c.text for c in self.chunks])), 256)
        self.assertGreater(len(self.chunks), len(self.originals))
        for original in self.originals:
            pieces = tuple(c for c in self.chunks if getattr(c, "original_chunk_id", c.chunk_id) == original.chunk_id)
            self.assertEqual(self.refiner.reconstruct(original, pieces), original.text)
            for piece in pieces:
                for f in fields(PolicyChunk):
                    if f.name not in {"text", "chunk_id", "source_start", "source_end"}:
                        self.assertEqual(getattr(piece, f.name), getattr(original, f.name), f.name)

    def test_unchanged_chunks_and_ids_remain_unchanged(self):
        counts = self.provider.token_counts([c.text for c in self.originals])
        for original, n in zip(self.originals, counts, strict=True):
            if n <= 256:
                self.assertTrue(any(c is original for c in self.chunks))
        again = self.refiner.refine_chunks(self.originals, self.provider).require_complete()
        self.assertEqual(self.chunks, again)
        self.assertEqual(len({c.chunk_id for c in self.chunks}), len(self.chunks))

    def test_all_sections_and_40_evidence_references_addressable(self):
        available = {(c.document_id, c.section_id) for c in self.chunks}
        for document in self.documents:
            headings = set(re.findall(r"(?m)^#{2,3} (\d+(?:\.\d+)*)\.? ", document.content))
            represented = {c.section_id for c in self.chunks if c.document_id == document.document_id}
            represented.update(c.parent_section for c in self.chunks if c.document_id == document.document_id and c.parent_section)
            self.assertEqual(headings, represented)
        refs = [r for q in json.loads((ROOT / "evaluations/questions.json").read_text())["questions"] for r in q["required_evidence"]]
        self.assertEqual(len(refs), 40)
        for ref in refs:
            self.assertIn((ref["document_id"], ref["section_id"]), available)

    def test_approved_mode_only_contains_approved_hr_chunks(self):
        originals = [c for d in load_documents(ROOT, mode="approved_only", reference_date=AS_OF)
                     for c in chunk_document(d).chunks]
        chunks = self.refiner.refine_chunks(originals, self.provider).require_complete()
        self.assertEqual({c.document_id for c in chunks}, {"HR-001"})
        self.assertEqual({c.status for c in chunks}, {"Approved (Synthetic)"})
        self.assertEqual(len({c.document_id for c in self.chunks}), 6)

    def test_plat_6_1_semantic_bundle_has_approval_and_full_scope_conditions(self):
        pieces = [c for c in self.chunks if c.document_id == "PLAT-001" and c.section_id == "6.1"]
        original = next(c for c in self.originals if c.document_id == "PLAT-001" and c.section_id == "6.1")
        approval = original.text[original.text.index("Record technical approval"):original.text.index("Bind the saved")]
        drift_and_conditions = original.text[original.text.index("If Terraform detects drift"):]
        bundle = next(c for c in pieces if "No separate Change Management approval" in c.text)
        self.assertIn(approval, bundle.text)
        self.assertIn(drift_and_conditions, bundle.text)
        self.assertLessEqual(self.provider.token_counts([bundle.text])[0], 256)
        self.assertEqual(bundle.status, "Draft — Proposed (Synthetic)")
        # A source link alone cannot satisfy these assertions: all three complete
        # paragraphs must physically be present in this single encoding input.
        self.assertTrue(any(s.role == "context" and "Record technical approval" in
                            bundle.text[s.text_start:s.text_end] for s in bundle.segments))
        self.assertEqual(set(bundle.required_context_ids), {c.chunk_id for c in pieces} - {bundle.chunk_id})

    def test_changed_plat_policy_requires_context_recipe_review(self):
        original = next(c for c in self.originals if c.document_id == "PLAT-001" and c.section_id == "6.1")
        result = self.refiner.refine_chunks([replace(original, content_hash="0" * 64)], self.provider)
        self.assertTrue(result.issues)
        self.assertIn("review", result.issues[0].reason.lower())

    def test_semantic_bundle_is_rejected_if_budget_cannot_hold_conditions(self):
        original = next(c for c in self.originals if c.document_id == "PLAT-001" and c.section_id == "6.1")
        smaller = SimpleNamespace(model=self.provider.model, max_input_tokens=240,
                                  token_counts=self.provider.token_counts)
        result = self.refiner.refine_chunks([original], smaller)
        self.assertEqual(result.chunks, ())
        self.assertIn("approval/scope context", result.issues[0].reason)

    def test_cli_reports_real_statistics_without_printing_policy_text_by_default(self):
        process = subprocess.run(
            [sys.executable, "-B", "-m", "northstar_atlas.ingestion.refiner",
             "--mode", "draft_review", "--section", "PLAT-001:6.1", "--samples", "2",
             "--reference-date", AS_OF.isoformat()],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("Original chunks: 151", process.stdout)
        self.assertIn("Oversized originals resolved: 25/25", process.stdout)
        self.assertIn("Unsafe originals: 0", process.stdout)
        self.assertIn("Encoded samples: 2; query vectors: 1; dimension: 384", process.stdout)
        self.assertNotIn("Record technical approval by an authorized", process.stdout)
        self.assertNotIn("Traceback", process.stderr)

    def test_real_refined_vectors_and_query_preserve_mapping_and_normalization(self):
        plat = [c for c in self.chunks if c.document_id == "PLAT-001" and c.section_id == "6.1"]
        table = next(c for c in self.chunks if c.document_id == "IT-001" and c.section_id == "3")
        samples = [*plat, table]
        results = embed_chunks(samples, self.provider)
        self.assertEqual(len(results), len(samples))
        for chunk, result in zip(samples, results, strict=True):
            self.assertIs(result.chunk, chunk)
            self.assertEqual(result.chunk_id, chunk.chunk_id)
            self.assertEqual(len(result.vector), 384)
            self.assertTrue(all(math.isfinite(x) for x in result.vector))
            self.assertAlmostEqual(math.hypot(*result.vector), 1.0, places=5)
            single = embed_chunks([chunk], self.provider)[0]
            self.assertLess(max(abs(a-b) for a,b in zip(result.vector,single.vector,strict=True)), 1e-5)
        query = embed_query("When does a regenerated Terraform plan require Change Management approval?", self.provider)
        self.assertEqual(query.model, results[0].model)
        self.assertEqual(len(query.vector), 384)
        self.assertAlmostEqual(math.hypot(*query.vector), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
