"""Chunking invariants, using verified corpus objects and in-memory fixtures."""

from dataclasses import FrozenInstanceError, replace
from datetime import date
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

from northstar_atlas.ingestion.document_loader import load_documents


ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 22)


class ChunkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)

    def setUp(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("northstar_atlas.ingestion.chunker"),
            "The section-aware chunker has not been implemented",
        )
        self.chunker = importlib.import_module("northstar_atlas.ingestion.chunker")

    def fixture(self, body: str):
        original = self.documents[0]
        header = original.content.split("## 1.", 1)[0]
        content = header + body
        return replace(original, content=content,
                       content_hash=hashlib.sha256(content.encode()).hexdigest())

    def assert_lossless(self, document, chunks) -> None:
        start = document.content.index("## ")
        self.assertEqual("".join(c.text for c in chunks), document.content[start:])
        cursor = start
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_index, index)
            self.assertEqual(chunk.source_start, cursor)
            self.assertEqual(chunk.text, document.content[chunk.source_start:chunk.source_end])
            cursor = chunk.source_end
        self.assertEqual(cursor, len(document.content))

    def test_all_six_produce_chunks_with_lossless_ordered_source_spans(self) -> None:
        self.assertEqual(len(self.documents), 6)
        for document in self.documents:
            with self.subTest(document=document.document_id):
                result = self.chunker.chunk_document(document)
                self.assertTrue(result.chunks)
                self.assert_lossless(document, result.chunks)
                self.assertNotIn("**Document ID:**", "".join(c.text for c in result.chunks))

    def test_numbered_ids_titles_and_parent_context_are_preserved(self) -> None:
        document = self.fixture(
            "## 4. Access\n\n### 4.2 Conditions\n\nOnly when approved.\n\n"
            "### 4.3 Removal\n\nRemove on expiry.\n\n## 5. Exceptions\n\nRecord them.\n"
        )
        chunks = self.chunker.chunk_document(document).chunks
        self.assertEqual([c.section_id for c in chunks], ["4.2", "4.3", "5"])
        self.assertEqual(chunks[0].parent_section, "4")
        self.assertEqual(chunks[0].parent_section_title, "Access")
        self.assertEqual(chunks[0].section_title, "Conditions")
        self.assertIsNone(chunks[-1].parent_section)
        self.assertIn("## 4. Access", chunks[0].text)
        self.assert_lossless(document, chunks)

    def test_corpus_heading_ids_and_evaluation_references_remain_addressable(self) -> None:
        available = set()
        for document in self.documents:
            chunks = self.chunker.chunk_document(document).chunks
            expected = set(re.findall(r"(?m)^#{2,3} (\d+(?:\.\d+)*)\.? ", document.content))
            actual = {c.section_id for c in chunks}
            actual.update(c.parent_section for c in chunks if c.parent_section)
            self.assertEqual(actual, expected)
            available.update((document.document_id, c.section_id) for c in chunks)
        questions = json.loads((ROOT / "evaluations/questions.json").read_text())["questions"]
        for question in questions:
            for ref in question["required_evidence"]:
                self.assertIn((ref["document_id"], ref["section_id"]), available)

    def test_ids_are_deterministic_and_unique_across_documents(self) -> None:
        first = [self.chunker.chunk_document(d) for d in self.documents]
        second = [self.chunker.chunk_document(d) for d in self.documents]
        self.assertEqual(first, second)
        ids = [c.chunk_id for result in first for c in result.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        document = self.fixture("## 1. Rule\n\nKeep this rule.\n")
        other = replace(document, document_id="OTHER-001")
        self.assertNotEqual(self.chunker.chunk_document(document).chunks[0].chunk_id,
                            self.chunker.chunk_document(other).chunks[0].chunk_id)

    def test_changed_hash_changes_every_chunk_id(self) -> None:
        document = self.documents[2]
        original = self.chunker.chunk_document(document).chunks
        changed = self.chunker.chunk_document(replace(document, content_hash="0" * 64)).chunks
        self.assertTrue(all(a.chunk_id != b.chunk_id for a, b in zip(original, changed, strict=True)))

    def test_every_source_metadata_field_is_copied_and_chunks_are_frozen(self) -> None:
        fields = ("document_id", "title", "department", "version", "status", "effective_date",
                  "source_path", "content_hash", "classification", "language", "organization",
                  "policy_owner", "review_cycle", "effective_date_text")
        for document in self.documents:
            for chunk in self.chunker.chunk_document(document).chunks:
                for field in fields:
                    self.assertEqual(getattr(chunk, field), getattr(document, field))
        with self.assertRaises(FrozenInstanceError):
            chunk.status = "Approved (Synthetic)"

    def test_modes_and_effective_date_filter_are_owned_by_loader(self) -> None:
        for mode, expected in (("approved_only", ["HR-001"]),
                               ("draft_review", [d.document_id for d in self.documents])):
            documents = load_documents(ROOT, mode=mode, reference_date=AS_OF)
            ids = [self.chunker.chunk_document(d).chunks[0].document_id for d in documents]
            self.assertEqual(ids, expected)
        self.assertEqual(load_documents(ROOT, reference_date=date(2026, 9, 30)), [])

    def test_citation_marks_proposed_status_and_preserves_section_version(self) -> None:
        document = self.documents[2]
        chunk = next(c for c in self.chunker.chunk_document(document).chunks if c.section_id == "6.1")
        self.assertEqual(chunk.citation, "PLAT-001 v0.1 §6.1 (proposed)")
        approved = self.chunker.chunk_document(self.documents[0]).chunks[0]
        self.assertEqual(approved.citation, "HR-001 v1.0 §1 (approved, synthetic)")

    def test_long_prose_splits_between_paragraphs_with_distinct_ids_same_citation(self) -> None:
        paragraphs = [f"Rule {i} is conditional. Keep its qualifier in this paragraph.\n\n" for i in range(8)]
        document = self.fixture("## 1. Scope\n\n### 1.1 Rules\n\n" + "".join(paragraphs))
        result = self.chunker.chunk_document(document, max_chars=160)
        self.assertGreater(len(result.chunks), 1)
        self.assertEqual(result.warnings, ())
        self.assertTrue(all(len(c.text) <= 160 for c in result.chunks))
        self.assertEqual({c.section_id for c in result.chunks}, {"1.1"})
        self.assertEqual({c.parent_section for c in result.chunks}, {"1"})
        self.assertEqual(len({c.citation for c in result.chunks}), 1)
        self.assertEqual(len({c.chunk_id for c in result.chunks}), len(result.chunks))
        for paragraph in paragraphs:
            self.assertEqual(sum(paragraph in c.text for c in result.chunks), 1)
        self.assert_lossless(document, result.chunks)
        wider = self.chunker.chunk_document(document, max_chars=300).chunks
        for a in result.chunks:
            for b in wider:
                if a.text != b.text:
                    self.assertNotEqual(a.chunk_id, b.chunk_id)

    def test_table_and_surrounding_conditions_stay_together_even_when_oversized(self) -> None:
        table = "| Role | Permission |\n| --- | --- |\n| IT | Provision |\n| Security | Review |\n"
        body = "## 2. Roles\n\nOnly after approval:\n\n" + table + "\nApproval does not confer execution authority.\n"
        document = self.fixture(body)
        result = self.chunker.chunk_document(document, max_chars=70)
        self.assertEqual(len(result.chunks), 1)
        self.assertIn(table, result.chunks[0].text)
        self.assertEqual(result.chunks[0].text, body)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("oversized atomic block", result.warnings[0])
        self.assertIn("§2", result.warnings[0])

    def test_lists_with_blank_lines_and_surrounding_qualifiers_remain_together(self) -> None:
        for marker in ("-", "1.", "1)"):
            body = (f"## 1. Requests\n\nOnly under these conditions:\n\n{marker} Obtain approval.\n\n"
                    f"{marker} Record execution.\n\nUnless revoked earlier, access expires at closure.\n")
            result = self.chunker.chunk_document(self.fixture(body), max_chars=80)
            self.assertEqual([c.text for c in result.chunks], [body])
            self.assertEqual(len(result.warnings), 1)

    def test_oversized_paragraph_is_retained_without_splitting_sentences(self) -> None:
        body = "## 1. Rule\n\n" + "A long condition " * 40 + "must remain whole.\n"
        result = self.chunker.chunk_document(self.fixture(body), max_chars=90)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, body)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("90", result.warnings[0])

    def test_actual_policy_qualifiers_remain_in_their_sections(self) -> None:
        for document in self.documents:
            result = self.chunker.chunk_document(document)
            # At the initial limit the corpus's substantive sections remain whole.
            own_ids = [c.section_id for c in result.chunks]
            self.assertEqual(len(own_ids), len(set(own_ids)))
        plat = self.chunker.chunk_document(self.documents[2]).chunks
        plan = next(c for c in plat if c.section_id == "6.1")
        self.assertIn("No separate Change Management approval", plan.text)
        self.assertIn("If\nthe result materially exceeds", plan.text)
        self.assertIn("stop and follow §4.3", plan.text)

    def test_crlf_unicode_and_trailing_whitespace_are_preserved(self) -> None:
        document = self.fixture("## 1. Règles\r\n\r\nKeep §1 — unchanged.  \r\n\r\n")
        self.assert_lossless(document, self.chunker.chunk_document(document).chunks)

    def test_fenced_content_does_not_create_false_headings_or_split(self) -> None:
        for fence in ("```", "~~~"):
            body = f"## 1. Example\n\n{fence}text\n## 9. Not a section\n\nretain\n{fence}\n\nLimits apply.\n"
            result = self.chunker.chunk_document(self.fixture(body), max_chars=40)
            self.assertEqual([c.section_id for c in result.chunks], ["1"])
            self.assertEqual(result.chunks[0].text, body)
            self.assertEqual(len(result.warnings), 1)

    def test_ambiguous_headings_and_unhandled_preamble_fail_explicitly(self) -> None:
        bodies = [
            "No numbered policy headings.\n",
            "## 1. One\n\nText.\n\n## 1. Duplicate\n\nText.\n",
            "### 1.1 Orphan\n\nText.\n",
            "## 1. One\n\n### 2.1 Wrong parent\n\nText.\n",
            "## 1. One\n\n## Unnumbered\n\nText.\n",
            "## 1. One\n\n#### 1.1.1 Unsupported depth\n\nText.\n",
            "Important rule before sections.\n\n## 1. One\n\nText.\n",
            "## 1. One\n\n```\nUnclosed example.\n",
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(self.chunker.ChunkingError):
                self.chunker.chunk_document(self.fixture(body))

    def test_invalid_limits_are_rejected(self) -> None:
        for limit in (0, -1, True, 2.5, "100"):
            with self.subTest(limit=limit), self.assertRaises(self.chunker.ChunkingError):
                self.chunker.chunk_document(self.documents[0], max_chars=limit)

    def run_cli(self, *args: str):
        return subprocess.run(
            [sys.executable, "-B", "-m", "northstar_atlas.ingestion.chunker",
             "--project-root", str(ROOT), *args],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )

    def test_cli_inspects_both_modes_sizes_sections_and_warnings_without_body(self) -> None:
        for mode, count in (("approved_only", 1), ("draft_review", 6)):
            result = self.run_cli("--mode", mode, "--reference-date", AS_OF.isoformat())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"Loaded {count} document(s)", result.stdout)
            self.assertIn("Total chunks:", result.stdout)
            self.assertIn("min/median/max", result.stdout)
            self.assertIn("§4.2", result.stdout)
            self.assertNotIn("This policy establishes procedures", result.stdout)
            if mode == "draft_review":
                self.assertIn("oversized atomic block", result.stderr)
        empty = self.run_cli("--reference-date", "2026-09-30")
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertIn("Total chunks: 0", empty.stdout)

    def test_cli_bad_arguments_and_missing_manifest_fail_without_traceback(self) -> None:
        for args in (("--max-chars", "0"), ("--reference-date", "20261022"),
                     ("--project-root", str(ROOT / ".local/nonexistent-m22-root"))):
            result = self.run_cli(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
