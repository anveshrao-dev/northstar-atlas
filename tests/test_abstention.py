"""Reviewed limitation evidence and bounded no-evidence replies; no network IO."""
from dataclasses import replace
import hashlib
import json
import unittest

from test_context import AS_OF, ROOT, chunk, family, hit
from test_grounding import saved_replay, terraform_obligations
from northstar_atlas import generation as g, grounding
from northstar_atlas.evidence import format_citation
from northstar_atlas.grounding import EvidenceObligation
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, load_documents
from northstar_atlas.llm import GenerationLimits, GenerationResponse, MODEL_ID


LIMITATION = "No numeric period is set here."
MISSING = "An approved retention duration for production change evidence."
NO_EVIDENCE = "The supplied evidence does not establish the requested answer."
LIVE = ROOT / ".local/m43-live/v2-smoke-20261023"


def retention_requirements(context):
    """Reviewed semantic limitation for the smoke test; quotations remain literal."""
    citation = next(group.citation for group in context.groups
                    if group.citation.document_id == "COMP-001" and group.citation.section_id == "11.1")
    return grounding.AbstentionRequirements(
        (EvidenceObligation("no_period_established", citation, LIMITATION, "evidence_linked"),),
        (MISSING,), "documented_gap")


def authored_retention(requirements):
    """A separately authored fixture, NOT a repaired or actual hosted response."""
    return {"kind": "abstain", "main_answer": {"claims": [], "obligations": {
        item.obligation_id: {"statement": item.source_quote,
                            "support": [{"citation": format_citation(item.citation), "quote": item.source_quote}],
                            "unresolved_reason": None} for item in requirements.limitations}},
        "missing_information": list(requirements.missing_information),
        "clarification_question": None, "reason": "insufficient_evidence"}


def no_evidence():
    return {"kind": "abstain", "main_answer": {"claims": [], "obligations": {}},
            "missing_information": [NO_EVIDENCE], "clarification_question": None,
            "reason": "insufficient_evidence"}


class CitedAbstentionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(grounding, "AbstentionRequirements"), "reviewed abstention contract missing")
        self.source = next(c for d in load_documents(ROOT, mode="draft_review") if d.document_id == "COMP-001"
                           for c in chunk_document(d).chunks if c.section_id == "11.1")
        self.context = self.prepare().context
        reviewed = retention_requirements(self.context)
        # Generic strict-validation tests explicitly request extractive mode;
        # the retention smoke requirement above permits a supported paraphrase.
        self.requirements = replace(reviewed, limitations=(
            replace(reviewed.limitations[0], validation_mode="extractive"),))

    def prepare(self, **kwargs):
        return g.prepare_answer("What is the approved duration?", [hit(self.source)],
                                mode="draft_review", reference_date=AS_OF, **kwargs)

    def validate(self, body, requirements=None):
        return g.validate_answer(json.dumps(body), self.context,
                                 abstention=self.requirements if requirements is None else requirements)

    def test_corrected_retention_renders_established_and_missing_with_proposed_citation(self):
        answer = self.validate(authored_retention(self.requirements))
        self.assertEqual(answer.kind, "abstain")
        self.assertEqual(answer.coverage.extractive_ids, ("no_period_established",))
        output = g.render_answer(answer)
        self.assertIn("What the supplied evidence establishes:", output)
        self.assertIn("Cannot establish from supplied evidence: " + MISSING, output)
        self.assertIn("Documented gap in supplied evidence", output)
        self.assertIn("COMP-001 v0.1 §11.1 (proposed)", output)
        self.assertIn(LIMITATION, output.split("Supporting quotations:")[0])
        self.assertEqual(answer.obligation_answers[0].support[0].match.kind, "exact")
        self.assertEqual(answer.abstention, self.requirements)

    def test_required_citation_and_support_cannot_be_omitted(self):
        for change in ("entry", "support", "citation"):
            body = authored_retention(self.requirements)
            entry = body["main_answer"]["obligations"]["no_period_established"]
            if change == "entry": body["main_answer"]["obligations"] = {}
            elif change == "support": entry["support"] = []
            else: del entry["support"][0]["citation"]
            with self.subTest(change=change), self.assertRaises(g.GenerationError): self.validate(body)

    def test_fabricated_or_changed_citation_and_quote_fail(self):
        for field, value in (("citation", "COMP-999 §11.1"),
                             ("citation", format_citation(self.requirements.limitations[0].citation).replace("(proposed)", "(approved, synthetic)")),
                             ("quote", "An approved period of seven years applies.")):
            body = authored_retention(self.requirements)
            body["main_answer"]["obligations"]["no_period_established"]["support"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(g.GenerationError): self.validate(body)

    def test_same_section_unrelated_quote_does_not_support_reviewed_limitation(self):
        body = authored_retention(self.requirements)
        body["main_answer"]["obligations"]["no_period_established"]["support"][0]["quote"] = "Do not silently select a period"
        with self.assertRaises(g.GenerationError): self.validate(body)

    def test_whitespace_normalization_retains_original_quote_and_source(self):
        quote = "No numeric period is set here. Future durations must be labeled proposed internal standards until approved;"
        body = authored_retention(self.requirements)
        body["main_answer"]["obligations"]["no_period_established"]["support"][0]["quote"] = quote
        support = self.validate(body).obligation_answers[0].support[0]
        self.assertEqual(support.match.kind, "whitespace_normalized")
        self.assertEqual(support.quote, quote)
        self.assertIn("\n", support.match.source_text)

    def test_invented_duration_in_statement_or_missing_information_fails(self):
        for duration in ("7 years", "seven years"):
            for field in ("statement", "missing_information"):
                body = authored_retention(self.requirements)
                value = "Retain production change evidence for " + duration + "."
                if field == "statement": body["main_answer"]["obligations"]["no_period_established"][field] = value
                else: body[field] = [value]
                with self.subTest(duration=duration, field=field), self.assertRaises(g.GenerationError): self.validate(body)

    def test_additional_even_cited_partial_policy_guidance_is_rejected(self):
        body = authored_retention(self.requirements)
        support = body["main_answer"]["obligations"]["no_period_established"]["support"]
        body["main_answer"]["claims"] = [{"text": "Retain everything forever.", "support": support}]
        with self.assertRaises(g.GenerationError): self.validate(body)

    def test_missing_unknown_duplicate_and_id_only_obligations_fail(self):
        for change in ("missing", "unknown", "id_only", "unresolved"):
            body = authored_retention(self.requirements)
            entries = body["main_answer"]["obligations"]
            if change == "missing": entries.clear()
            elif change == "unknown": entries["invented"] = entries.pop("no_period_established")
            elif change == "id_only": entries["no_period_established"]["statement"] = "no_period_established"
            else: entries["no_period_established"].update(statement=None, support=[], unresolved_reason="Unknown")
            with self.subTest(change=change), self.assertRaises(g.GenerationError): self.validate(body)
        body = authored_retention(self.requirements)
        entry = json.dumps(body["main_answer"]["obligations"]["no_period_established"])
        raw = json.dumps(body).replace('"no_period_established": ' + entry,
                                       '"no_period_established": ' + entry + ', "no_period_established": ' + entry)
        with self.assertRaises(g.GenerationError): g.validate_answer(raw, self.context, abstention=self.requirements)

    def test_contract_is_specific_to_abstention_and_no_unresolved_reason_is_rendered(self):
        for field, value in (("kind", "answer"), ("reason", "conflicting_evidence"),
                             ("clarification_question", "What duration?")):
            body = authored_retention(self.requirements); body[field] = value
            with self.subTest(field=field), self.assertRaises(g.GenerationError): self.validate(body)

    def test_reviewed_evidence_must_be_current_present_and_eligible(self):
        item = self.requirements.limitations[0]
        for citation in (replace(item.citation, content_hash="f" * 64), replace(item.citation, version="9.9"),
                         replace(item.citation, status=APPROVED), replace(item.citation, chunk_ids=("missing",))):
            invalid = replace(self.requirements, limitations=(replace(item, citation=citation),))
            with self.subTest(citation=citation), self.assertRaises(g.GenerationError): self.prepare(abstention=invalid)
        with self.assertRaises(g.GenerationError):
            g.prepare_answer("When?", [hit(self.source)], mode="approved_only", reference_date=AS_OF, abstention=self.requirements)
        with self.assertRaises(g.GenerationError):
            g.prepare_answer("When?", [], reference_date=AS_OF, abstention=self.requirements)
        with self.assertRaises(g.GenerationError):
            self.prepare(abstention=replace(self.requirements, limitations=(replace(item, source_quote="Invented gap."),)))

    def test_approved_effective_evidence_can_support_abstention(self):
        source = chunk("The schedule is not defined here.\n", doc="HR-001", status=APPROVED, effective=AS_OF)
        base = g.prepare_answer("When?", [hit(source)], reference_date=AS_OF)
        requirement = grounding.AbstentionRequirements((EvidenceObligation("schedule_gap", base.context.groups[0].citation,
                    "The schedule is not defined here.", "extractive"),), ("An approved schedule.",), "documented_gap")
        prepared = g.prepare_answer("When?", [hit(source)], reference_date=AS_OF, abstention=requirement)
        answer = g.validate_answer(json.dumps(authored_retention(requirement)), prepared.context, abstention=requirement)
        self.assertIn("(approved, synthetic)", g.render_answer(answer))
        with self.assertRaises(g.GenerationError):
            g.prepare_answer("When?", [hit(source)], reference_date=AS_OF.replace(year=2025), abstention=requirement)

    def test_contract_fields_and_mixed_operational_requirements_are_rejected(self):
        for invalid in (replace(self.requirements, limitations=()), replace(self.requirements, missing_information=()),
                        replace(self.requirements, missing_information=(" ",)), replace(self.requirements, missing_information=(MISSING, MISSING)),
                        replace(self.requirements, missing_information=[MISSING]), replace(self.requirements, basis="universal_absence")):
            with self.subTest(invalid=invalid), self.assertRaises(g.GenerationError): self.prepare(abstention=invalid)
        with self.assertRaises(g.GenerationError): self.prepare(abstention=self.requirements, obligations=self.requirements.limitations)

    def test_schema_prompt_and_budget_include_reviewed_contract(self):
        p = self.prepare(abstention=self.requirements)
        props = p.request.schema["properties"]
        self.assertEqual(props["kind"]["enum"], ["abstain"])
        self.assertEqual(props["main_answer"]["properties"]["claims"]["maxItems"], 0)
        entry = props["main_answer"]["properties"]["obligations"]["properties"]["no_period_established"]
        self.assertEqual(entry["properties"]["support"]["minItems"], 1)
        self.assertEqual(props["missing_information"]["items"]["enum"], [MISSING])
        self.assertEqual(p.abstention, self.requirements)
        self.assertEqual(json.loads(p.request.user)["reviewed_abstention"]["basis"], "documented_gap")
        self.assertGreater(p.request.input_token_bound, self.prepare().request.input_token_bound)
        self.assertEqual(p, self.prepare(abstention=self.requirements))
        self.assertEqual(g.SCHEMA_VERSION, "grounded-answer-v2")
        with self.assertRaises(g.GenerationError): self.prepare(abstention=self.requirements,
            limits=GenerationLimits(context_tokens=1024, output_tokens=128, margin_tokens=64))

    def test_provider_roundtrip_and_retrieval_forward_contract(self):
        requirements = self.requirements
        class Store:
            def search(inner, *args, **kwargs): return (hit(self.source),)
        prepared = g.retrieve_context("When?", Store(), object(), corpus_fingerprint="snapshot", mode="draft_review",
                                      reference_date=AS_OF, abstention=requirements)
        class Fake:
            def generate(inner, request):
                return GenerationResponse(json.dumps(authored_retention(requirements)), MODEL_ID, "stop", None, 0, 1)
        self.assertEqual(g.generate_answer(prepared, Fake()).answer.kind, "abstain")

    def test_missing_required_sibling_context_prevents_preparation(self):
        parts = family(self.source, [50])
        with self.assertRaises(g.GenerationError):
            g.prepare_answer("When?", [hit(parts[0])], mode="draft_review", reference_date=AS_OF, abstention=self.requirements)

    def test_empty_context_fast_path_cannot_bypass_reviewed_requirements(self):
        class Never:
            def generate(inner, request): raise AssertionError("provider called")
        prepared = g.prepare_answer("When?", [], reference_date=AS_OF)
        with self.assertRaises(g.GenerationError): g.generate_answer(replace(prepared, abstention=self.requirements), Never())

    def test_evidence_linked_is_explicit_and_does_not_require_exact_wording(self):
        requirement = replace(self.requirements, limitations=(replace(self.requirements.limitations[0], validation_mode="evidence_linked"),))
        body = authored_retention(requirement)
        body["main_answer"]["obligations"]["no_period_established"]["statement"] = "This draft supplies no numeric duration."
        answer = self.validate(body, requirement)
        self.assertEqual(answer.coverage.evidence_linked_ids, ("no_period_established",))
        self.assertIn("require semantic review", g.render_answer(answer))
        with self.assertRaises(g.GenerationError): self.validate(body)

    def test_documented_gap_and_retrieval_insufficiency_are_not_conflated(self):
        requirement = replace(self.requirements, basis="insufficient_evidence")
        output = g.render_answer(self.validate(authored_retention(requirement), requirement))
        self.assertIn("Insufficient supplied evidence", output)
        self.assertNotIn("Documented gap", output)

    def test_retention_smoke_mode_accepts_limitation_paraphrase_but_keeps_literal_support(self):
        reviewed = retention_requirements(self.context)
        self.assertEqual(reviewed.limitations[0].validation_mode, "evidence_linked")
        body = authored_retention(reviewed)
        entry = body["main_answer"]["obligations"]["no_period_established"]
        entry["statement"] = "No numeric period is set for production change evidence retention in the proposed policy."
        answer = self.validate(body, reviewed)
        self.assertEqual(answer.coverage.evidence_linked_ids, ("no_period_established",))
        self.assertEqual(answer.coverage.extractive_ids, ())
        self.assertIn("require semantic review", g.render_answer(answer))
        strict = replace(reviewed, limitations=(replace(reviewed.limitations[0], validation_mode="extractive"),))
        with self.assertRaises(g.IncompleteAnswerError): self.validate(body, strict)
        for field, value in (("citation", "COMP-999 §11.1"), ("quote", "No retention obligation exists anywhere.")):
            changed = json.loads(json.dumps(body))
            changed["main_answer"]["obligations"]["no_period_established"]["support"][0][field] = value
            with self.subTest(field=field), self.assertRaises(g.GenerationError): self.validate(changed, reviewed)
        body["missing_information"] = ["Retain records for seven years."]
        with self.assertRaises(g.GenerationError): self.validate(body, reviewed)

    @unittest.skipUnless((ROOT / ".local/m43-live/cited-retention-20261023/raw-response-1.json").is_file(),
                         "optional historical live artifact")
    def test_latest_retention_replay_passes_linked_mode_without_changing_response(self):
        folder = ROOT / ".local/m43-live/cited-retention-20261023"
        path = folder / "raw-response-1.json"
        original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        context, raw = saved_replay(1, folder)
        requirements = retention_requirements(context)
        answer = g.validate_answer(raw, context, abstention=requirements)
        self.assertEqual(answer.kind, "abstain")
        self.assertEqual(answer.coverage.evidence_linked_ids, ("no_period_established",))
        self.assertEqual(answer.coverage.missing_ids, ())
        entry = answer.obligation_answers[0]
        self.assertEqual(entry.support[0].match.kind, "exact")
        self.assertEqual(entry.support[0].quote, LIMITATION)
        self.assertEqual(entry.support[0].citation.status, "Draft — Proposed (Synthetic)")
        self.assertEqual(answer.missing_information, (MISSING,))
        self.assertEqual(answer.abstention.basis, "documented_gap")
        strict = replace(requirements, limitations=(replace(requirements.limitations[0], validation_mode="extractive"),))
        with self.assertRaises(g.IncompleteAnswerError): g.validate_answer(raw, context, abstention=strict)
        self.assertEqual(original_hash, hashlib.sha256(path.read_bytes()).hexdigest())

    @unittest.skipUnless((LIVE / "raw-response-2.json").is_file(), "optional historical live artifact")
    def test_saved_retention_rejected_and_authored_correction_passes_without_editing_history(self):
        paths = tuple(LIVE.glob("*.json"))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        context, raw = saved_replay(2, LIVE)
        requirements = retention_requirements(context)
        with self.assertRaises(g.GenerationError): g.validate_answer(raw, context, abstention=requirements)
        corrected = g.validate_answer(json.dumps(authored_retention(requirements)), context, abstention=requirements)
        self.assertEqual(corrected.coverage.satisfied_ids, ("no_period_established",))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

    @unittest.skipUnless((LIVE / "raw-response-1.json").is_file(), "optional historical live artifact")
    def test_saved_terraform_still_passes_all_four_obligations(self):
        context, raw = saved_replay(1, LIVE)
        answer = g.validate_answer(raw, context, obligations=terraform_obligations(context))
        self.assertEqual(len(answer.coverage.extractive_ids), 4)
        self.assertEqual(answer.coverage.missing_ids, ())
        self.assertEqual(answer.kind, "answer")


class NoEvidenceAbstentionTests(unittest.TestCase):
    def test_free_prose_cannot_bypass_grounding_in_missing_information(self):
        context = g.prepare_answer("When?", [], reference_date=AS_OF).context
        for value in ("Retain records for 7 years.", "The policy mandates indefinite retention.", "No policy exists anywhere."):
            body = no_evidence(); body["missing_information"] = [value]
            with self.subTest(value=value), self.assertRaises(g.GenerationError): g.validate_answer(json.dumps(body), context)

    def test_bounded_no_evidence_abstention_without_citations(self):
        for hits in ([], [hit(chunk())]):
            context = g.prepare_answer("When?", hits, mode="draft_review", reference_date=AS_OF).context
            answer = g.validate_answer(json.dumps(no_evidence()), context)
            output = g.render_answer(answer)
            self.assertIn(NO_EVIDENCE, output)
            self.assertNotIn("Documented gap", output)
            self.assertNotIn("Supporting quotations:", output)
            self.assertFalse(answer.claims)

    def test_empty_context_does_not_call_provider_or_infer_gap(self):
        prepared = g.prepare_answer("When?", [], reference_date=AS_OF)
        class Never:
            def generate(self, request): raise AssertionError("provider called")
        result = g.generate_answer(prepared, Never())
        self.assertEqual(result.answer.missing_information, (NO_EVIDENCE,))
        self.assertIsNone(result.response)
        self.assertNotIn("Documented gap", g.render_answer(result.answer))

    def test_no_evidence_cannot_contain_cited_partial_claims(self):
        context = g.prepare_answer("When?", [hit(chunk())], mode="draft_review", reference_date=AS_OF).context
        body = no_evidence()
        body["main_answer"]["claims"] = [{"text": "Proceed without review.", "support": [{
            "citation": format_citation(context.groups[0].citation), "quote": "Only within the approved scope."}]}]
        with self.assertRaises(g.GenerationError): g.validate_answer(json.dumps(body), context)

    def test_unresolved_operational_reason_is_not_published_as_guidance(self):
        context = g.prepare_answer("Who?", [hit(chunk())], mode="draft_review", reference_date=AS_OF).context
        item = EvidenceObligation("scope", context.groups[0].citation, "Only within the approved scope.", "extractive")
        body = no_evidence()
        body["main_answer"]["obligations"] = {"scope": {"statement": None, "support": [],
                                                        "unresolved_reason": "Delete every record now."}}
        answer = g.validate_answer(json.dumps(body), context, obligations=(item,))
        self.assertNotIn("Delete every record", g.render_answer(answer))
        self.assertIn("Missing reviewed statements: scope", g.render_answer(answer))
