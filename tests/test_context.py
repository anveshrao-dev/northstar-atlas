"""Offline contracts for evidence assembly; fixture counters are not LLM tokenizers."""

from dataclasses import fields, replace
from datetime import date
import importlib
import importlib.util
import json
from pathlib import Path
import unittest

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.ingestion.chunker import PolicyChunk, chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, load_documents
from northstar_atlas.ingestion.refiner import RefinedPolicyChunk, TextSegment
from northstar_atlas.vector_store.contracts import SearchHit


ROOT = Path(__file__).resolve().parents[1]
AS_OF = date(2026, 10, 22)
MODEL = EmbeddingModel("fixture", "r1", 3, "l2")


class CharacterCounter:
    """One character plus two wrapper units; exact test arithmetic, not model tokens."""

    name = "fixture-characters-plus-two"

    def count_tokens(self, text: str) -> int:
        return len(text) + 2


def chunk(text="Only within the approved scope.\n", *, key="one", doc="IT-001", section="4.1",
          status=DRAFT, effective=None, start=100):
    return PolicyChunk(
        chunk_id=key, document_id=doc, title="Synthetic Policy", department="IT Operations",
        version="0.1" if status == DRAFT else "1.0", status=status, effective_date=effective,
        source_path=Path(f"data/policies/{doc}.md"), content_hash="a" * 64,
        section_id=section, section_title="Controls", parent_section=section.split('.')[0],
        text=text, chunk_index=0, parent_section_title="Access", source_start=start,
        source_end=start + len(text), classification="Internal", language="English",
        organization="Northstar Technologies", policy_owner="Policy Owner",
        review_cycle="Annual", effective_date_text=effective.isoformat() if effective else "Pending approval",
    )


def family(original, cuts):
    """Hand-built fragments: full primary partition plus repeated first-line context."""
    pieces = []
    prefix = original.text.splitlines(keepends=True)[0]
    edges = [0, *cuts, len(original.text)]
    keys = tuple(f"{original.chunk_id}-part-{i}" for i in range(len(edges) - 1))
    for i, (a, b) in enumerate(zip(edges, edges[1:])):
        primary = original.text[a:b]
        text = primary if i == 0 else prefix + primary
        spans = (() if i == 0 else (TextSegment(original.source_start,
                 original.source_start + len(prefix), 0, len(prefix), "context"),))
        offset = 0 if i == 0 else len(prefix)
        spans += (TextSegment(original.source_start + a, original.source_start + b,
                              offset, len(text), "source"),)
        data = {f.name: getattr(original, f.name) for f in fields(PolicyChunk)}
        data.update(chunk_id=keys[i], text=text, source_start=original.source_start + a,
                    source_end=original.source_start + b)
        pieces.append(RefinedPolicyChunk(**data, original_chunk_id=original.chunk_id,
                      refinement_index=i, refinement_count=len(keys), segments=spans,
                      required_context_ids=tuple(k for k in keys if k != keys[i]),
                      refinement_version="token-context-v1", embedding_model=MODEL, token_limit=256))
    return tuple(pieces)


def hit(c, siblings=()):
    return SearchHit(c, 0.2, MODEL, tuple(siblings))


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("northstar_atlas.context"),
                             "M4.2 context builder is not implemented")
        self.context = importlib.import_module("northstar_atlas.context")
        self.evidence = importlib.import_module("northstar_atlas.evidence")
        self.counter = CharacterCounter()

    def build(self, hits, **kwargs):
        options = dict(counter=self.counter, max_tokens=100_000, mode="draft_review", reference_date=AS_OF)
        options.update(kwargs)
        return self.context.build_context(hits, **options)

    def test_citation_formats_version_section_and_proposed_status(self):
        package = self.build([hit(chunk(doc="PLAT-001", section="6.1"))])
        ref = package.groups[0].citation
        self.assertEqual(ref.label, "PLAT-001 v0.1 §6.1 (proposed)")
        marker = self.evidence.format_citation(ref)
        self.assertTrue(self.evidence.citation_syntax_valid(marker))
        self.assertEqual(self.evidence.validate_citations([marker, ref], package), (ref,))

    def test_citation_membership_rejects_invented_or_changed_identity(self):
        package = self.build([hit(chunk())])
        ref = package.groups[0].citation
        for changed in (replace(ref, document_id="FAKE-999"), replace(ref, section_id="99.9"),
                        replace(ref, version="9.0"), replace(ref, status=APPROVED),
                        replace(ref, content_hash="b" * 64), replace(ref, chunk_ids=("invented",))):
            with self.subTest(changed=changed):
                with self.assertRaises(self.evidence.CitationError):
                    self.evidence.validate_citations([changed], package)
        invented = self.evidence.format_citation(ref).replace("IT-001", "FAKE-999")
        self.assertTrue(self.evidence.citation_syntax_valid(invented))
        with self.assertRaises(self.evidence.CitationError):
            self.evidence.validate_citations([invented], package)

    def test_malformed_citation_is_not_accepted_as_a_document_match(self):
        package = self.build([hit(chunk())])
        for text in ("IT-001 §4.1", "", "[E1] IT-001 v0.1 §4.1 (proposed)",
                     self.evidence.format_citation(package.groups[0].citation) + " trailing"):
            self.assertFalse(self.evidence.citation_syntax_valid(text))
            with self.assertRaises(self.evidence.CitationError):
                self.evidence.validate_citations([text], package)

    def test_citation_is_stable_when_rank_and_budget_change(self):
        a, b = chunk(), chunk(key="two", doc="SEC-001")
        first = self.build([hit(a), hit(b)])
        second = self.build([hit(b), hit(a)])
        self.assertEqual(first.groups[0].citation, second.groups[1].citation)
        changed = self.build([hit(replace(a, content_hash="b" * 64))])
        self.assertNotEqual(first.groups[0].citation.reference_id, changed.groups[0].citation.reference_id)

    def test_counts_entire_rendered_payload_including_metadata_and_wrappers(self):
        package = self.build([hit(chunk())])
        self.assertEqual(package.token_count, len(package.rendered) + 2)
        self.assertGreater(package.token_count, len(chunk().text) + 2)
        self.assertEqual(package.token_counter, self.counter.name)
        same = self.build([hit(chunk())], max_tokens=package.token_count)
        self.assertEqual(same.rendered, package.rendered)
        self.assertEqual(len(same.groups), 1)

    def test_budget_excludes_complete_group_and_rejects_its_citation(self):
        a, b = chunk(), chunk(key="two", doc="SEC-001", text="Keep all qualifications.\n" * 30)
        full = self.build([hit(a), hit(b)])
        first = self.build([hit(a)])
        reduced = self.build([hit(a), hit(b)], max_tokens=first.token_count)
        self.assertEqual([g.citation.document_id for g in reduced.groups], ["IT-001"])
        self.assertTrue(any(w.code == "budget_excluded" and "two" in w.chunk_ids for w in reduced.warnings))
        self.assertLessEqual(reduced.token_count, reduced.max_tokens)
        with self.assertRaises(self.evidence.CitationError):
            self.evidence.validate_citations([full.groups[1].citation], reduced)

    def test_oversized_group_is_reported_without_partial_text(self):
        original = chunk("All conditions apply.\n" + "Require technical approval.\n" * 10)
        parts = family(original, [len("All conditions apply.\n")])
        empty = self.build([])
        package = self.build([hit(parts[0], parts[1:])], max_tokens=empty.token_count)
        self.assertFalse(package.groups)
        self.assertNotIn("Require technical approval", package.rendered)
        self.assertTrue(any(w.code == "budget_excluded" and set(w.chunk_ids) == {p.chunk_id for p in parts}
                            for w in package.warnings))

    def test_unusable_envelope_budget_is_an_error(self):
        with self.assertRaises(self.evidence.ContextError):
            self.build([], max_tokens=1)

    def test_invalid_counter_results_or_errors_fail_closed(self):
        for value in (True, -1, 0, 1.5, "2"):
            class BadCounter(CharacterCounter):
                def count_tokens(self, text): return value
            with self.subTest(value=value), self.assertRaises(self.evidence.ContextError):
                self.build([], counter=BadCounter())
        class BrokenCounter(CharacterCounter):
            def count_tokens(self, text): raise RuntimeError("private detail")
        with self.assertRaises(self.evidence.ContextError) as caught:
            self.build([], counter=BrokenCounter())
        self.assertNotIn("private detail", str(caught.exception))

    def test_duplicate_hits_and_siblings_do_not_duplicate_source_text(self):
        original = chunk("All conditions apply.\nFirst approval.\nSecond approval.\n")
        pieces = family(original, [len("All conditions apply.\nFirst approval.\n")])
        p = self.build([hit(pieces[0], pieces[1:]), hit(pieces[0], pieces[1:]), hit(pieces[1], pieces[:1])])
        self.assertEqual(len(p.groups), 1)
        group = p.groups[0]
        self.assertEqual(group.text, original.text)
        self.assertEqual(len(group.sources), 2)
        self.assertEqual(group.duplicate_characters_removed, len("All conditions apply.\n"))
        self.assertEqual([s.direct_rank for s in group.sources], [1, 3])
        self.assertEqual(group.sources[0].chunk, pieces[0])

    def test_same_text_at_distinct_offsets_or_documents_is_not_removed(self):
        a = chunk("Same rule.\n", key="a", start=100)
        b = replace(a, chunk_id="b", source_start=111, source_end=122, chunk_index=1)
        c = chunk("Same rule.\n", key="c", doc="SEC-001")
        p = self.build([hit(a), hit(b), hit(c)])
        self.assertEqual(p.groups[0].text, "Same rule.\nSame rule.\n")
        self.assertEqual(p.groups[1].text, "Same rule.\n")

    def test_siblings_keep_tables_lists_and_conditions_in_source_order(self):
        text = "Only within approved scope.\n\n| Owner | Decision |\n| --- | --- |\n| Platform | Technical approval |\n\n- Keep the window.\n- Stop if risk changes.\n"
        original = chunk(text)
        pieces = family(original, [text.index("| Owner"), text.index("- Keep")])
        p = self.build([hit(pieces[2], reversed(pieces[:2]))])
        self.assertEqual(p.groups[0].text, text)
        self.assertEqual([s.chunk.chunk_id for s in p.groups[0].sources], [c.chunk_id for c in pieces])
        self.assertEqual([s.direct_rank for s in p.groups[0].sources], [None, None, 1])

    def test_transitive_missing_sibling_is_a_critical_error(self):
        original = chunk("Intro.\nFirst.\nSecond.\n")
        pieces = family(original, [7, 14])
        with self.assertRaises(self.evidence.ContextError) as caught:
            self.build([hit(pieces[0], [pieces[1]])])
        self.assertTrue(any(w.code == "missing_sibling" and pieces[2].chunk_id in w.chunk_ids
                            for w in caught.exception.warnings))

    def test_required_sibling_can_be_supplied_as_another_direct_hit(self):
        original = chunk("Intro.\nFirst.\nSecond.\n")
        pieces = family(original, [14])
        p = self.build([hit(pieces[0]), hit(pieces[1])])
        self.assertEqual(p.groups[0].text, original.text)

    def test_sibling_identity_mismatch_is_rejected(self):
        pieces = family(chunk("Intro.\nFirst.\nSecond.\n"), [14])
        for bad in (replace(pieces[1], content_hash="b" * 64), replace(pieces[1], section_id="5.1"),
                    replace(pieces[1], status=APPROVED, effective_date=AS_OF)):
            with self.subTest(bad=bad), self.assertRaises(self.evidence.ContextError):
                self.build([hit(pieces[0], [bad])])

    def test_conflicting_duplicate_id_and_overlapping_content_are_rejected(self):
        a = chunk("ABCDE", key="a")
        for b in (replace(a, text="WRONG"), replace(a, chunk_id="b", text="WRONG")):
            with self.subTest(b=b), self.assertRaises(self.evidence.ContextError):
                self.build([hit(a), hit(b)])

    def test_untracked_refined_text_and_invalid_spans_are_rejected(self):
        pieces = family(chunk("Intro.\nFirst.\nSecond.\n"), [14])
        for bad in (replace(pieces[1], text=pieces[1].text + "untracked rule"),
                    replace(pieces[1], segments=(replace(pieces[1].segments[0], source_end=900), *pieces[1].segments[1:]))):
            with self.subTest(bad=bad), self.assertRaises(self.evidence.ContextError):
                self.build([hit(pieces[0], [bad])])

    def test_same_document_cannot_mix_source_snapshots(self):
        a = chunk(key="a")
        b = replace(a, chunk_id="b", section_id="5.1", content_hash="b" * 64)
        with self.assertRaises(self.evidence.ContextError):
            self.build([hit(a), hit(b)])

    def test_disconnected_excerpts_have_explicit_gap_warning(self):
        a = chunk("Before.\n", key="a")
        b = chunk("After.\n", key="b", start=200)
        package = self.build([hit(a), hit(b)])
        self.assertEqual(len(package.groups[0].spans), 2)
        self.assertTrue(any(w.code == "source_gap" for w in package.warnings))
        self.assertNotEqual(package.groups[0].text, "Before.\nAfter.\n")

    def test_approved_only_excludes_drafts_and_future_approved_policies(self):
        approved = chunk(doc="HR-001", status=APPROVED, effective=AS_OF)
        future = chunk(key="future", doc="SEC-001", status=APPROVED, effective=date(2099, 1, 1))
        p = self.build([hit(chunk(key="draft")), hit(approved), hit(future)], mode="approved_only")
        self.assertEqual([g.citation.document_id for g in p.groups], ["HR-001"])
        body = json.loads(p.rendered)
        self.assertEqual(len(body["approved_evidence"]), 1)
        self.assertEqual(body["proposed_evidence"], [])
        self.assertEqual(len([w for w in p.warnings if w.code == "ineligible"]), 2)

    def test_draft_review_keeps_authority_separate(self):
        approved = chunk(key="hr", doc="HR-001", status=APPROVED, effective=AS_OF)
        p = self.build([hit(chunk()), hit(approved)])
        body = json.loads(p.rendered)
        self.assertEqual(body["approved_evidence"][0]["status"], APPROVED)
        self.assertEqual(body["proposed_evidence"][0]["status"], DRAFT)
        self.assertFalse(body["proposed_evidence"][0]["effective_on_reference_date"])

    def test_ineligible_sibling_of_approved_hit_is_not_silently_dropped(self):
        original = chunk("Intro.\nFirst.\nSecond.\n", doc="HR-001", status=APPROVED, effective=AS_OF)
        pieces = family(original, [14])
        bad = replace(pieces[1], status=DRAFT, effective_date=None)
        with self.assertRaises(self.evidence.ContextError):
            self.build([hit(pieces[0], [bad])], mode="approved_only")

    def test_unrelated_context_does_not_enter_the_evidence_package(self):
        p = self.build([hit(chunk(), [chunk(doc="SEC-001", key="unrelated")])])
        self.assertEqual(len(p.groups), 1)
        self.assertNotIn("SEC-001", p.rendered)
        self.assertTrue(any(w.code == "unrelated_context" for w in p.warnings))

    def test_empty_results_have_no_citations_and_explicit_warning(self):
        package = self.build([])
        self.assertFalse(package.groups)
        self.assertTrue(any(w.code == "empty_results" for w in package.warnings))
        self.assertEqual(self.evidence.validate_citations([], package), ())

    def test_deterministic_output_despite_sibling_input_order(self):
        pieces = family(chunk("Intro.\nFirst.\nSecond.\n"), [7, 14])
        a = self.build([hit(pieces[0], pieces[1:])])
        b = self.build([hit(pieces[0], reversed(pieces[1:]))])
        self.assertEqual(a, b)

    def test_invalid_arguments_status_dates_and_paths_are_rejected(self):
        for kwargs in ({"max_tokens": True}, {"max_tokens": 0}, {"mode": "all"}, {"reference_date": "today"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(self.evidence.ContextError):
                self.build([], **kwargs)
        for bad in (replace(chunk(), status="Approved"), replace(chunk(), effective_date=AS_OF),
                    replace(chunk(), source_path=Path("/tmp/policy.md")),
                    replace(chunk(), source_path=Path("../policy.md"))):
            with self.subTest(bad=bad), self.assertRaises(self.evidence.ContextError):
                self.build([hit(bad)])

    def test_real_terraform_section_conditions_remain_one_citable_group(self):
        documents = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        plat = next(d for d in documents if d.document_id == "PLAT-001")
        original = next(c for c in chunk_document(plat).chunks if c.section_id == "6.1")
        # Fake retrieval fixture using real verified source; no embedding/model/network work.
        split = original.text.index("No separate Change Management approval")
        pieces = family(original, [split])
        p = self.build([hit(pieces[0], pieces[1:])])
        self.assertEqual(len(p.groups), 1)
        self.assertEqual(p.groups[0].text, original.text)
        for condition in ("Record technical approval by an authorized Platform reviewer",
                          "it is not\nthe Terraform technical approver",
                          "within the authorized CR scope, assessed risk,\ndeployment window",
                          "stop and follow §4.3's amendment or escalation process"):
            self.assertIn(condition, p.groups[0].text)
        self.assertEqual(p.groups[0].citation.label, "PLAT-001 v0.1 §6.1 (proposed)")
        excluded = self.build([hit(pieces[0], pieces[1:])], max_tokens=self.build([]).token_count)
        self.assertFalse(excluded.groups)
        with self.assertRaises(self.evidence.CitationError):
            self.evidence.validate_citations([p.groups[0].citation], excluded)


if __name__ == "__main__":
    unittest.main()
