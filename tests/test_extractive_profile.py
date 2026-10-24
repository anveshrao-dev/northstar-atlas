"""Application-owned excerpts; fixture evidence does not establish retrieval quality."""
from dataclasses import asdict, replace
import json
import socket
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, chunk, family, hit
from test_grounding import package, terraform_obligations
from northstar_atlas import generation as g, grounding as gr
from northstar_atlas.evidence import EvidenceSpan
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, load_documents
from northstar_atlas.llm import GenerationLimits, GenerationResponse, MODEL_ID
from test_qa import FakeLLM, FixtureStore, authored_claims
from test_generation_v2 import payload
from test_abstention import authored_retention, retention_requirements


QUESTION = "Does regenerating a Terraform plan require another Change Management approval?"


def terraform_scope():
    """Reviewed fixture offsets, NOT automatic runtime applicability selection."""
    return gr.ReviewedExtractiveScope(QUESTION, "draft_review", tuple(
        gr.ReviewedExtractiveSpan(key, start, end) for key, start, end in (
            ("renewed_technical_approval", 18600, 18790),
            ("no_automatic_change_approval", 18964, 19053),
            ("authorized_bounds", 19054, 19253),
            ("stop_and_amend", 19278, 19422))))


def refs_payload(obligations):
    body = payload()
    body["main_answer"]["obligations"] = {
        o.obligation_id: {"action": "include_reviewed"} if o.validation_mode == "extractive"
        else payload((o,))["main_answer"]["obligations"][o.obligation_id] for o in obligations}
    return body


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(gr, "ReviewedExtractiveScope"), "reviewed span binding API is missing")
        self.p = package()
        self.ob = gr.EvidenceObligation("review", self.p.groups[0].citation,
                                       "A reviewer must review and approve the plan.", "extractive")
        self.match = gr.match_quotation(self.ob.source_quote, self.p.groups[0])
        self.span = gr.ReviewedExtractiveSpan("review", self.match.source_start, self.match.source_end)
        self.scope = gr.ReviewedExtractiveScope("What is required?", "draft_review", (self.span,))

    def bind(self, obligations=None, scope=None, context=None):
        return gr.bind_extractive_obligations(
            (self.ob,) if obligations is None else obligations,
            self.p if context is None else context, self.scope if scope is None else scope,
            question="What is required?")

    def test_binds_exact_source_and_preserves_original_reviewed_quote(self):
        bound, = self.bind()
        self.assertEqual(bound.obligation, self.ob)
        self.assertEqual(bound.match.source_text, "A reviewer must\nreview and approve the plan.")
        self.assertEqual(bound.match.kind, "whitespace_normalized")
        self.assertEqual(bound.match.source_start, 100)
        self.assertEqual(bound.match.source_end, 144)
        self.assertEqual(bound.obligation.source_quote, self.ob.source_quote)

    def test_exact_quote_binding_remains_exact(self):
        ob = replace(self.ob, source_quote=self.match.source_text)
        self.assertEqual(self.bind((ob,))[0].match.kind, "exact")

    def test_incorrect_span_bounds_types_and_identity_are_rejected(self):
        for changed in (replace(self.span, source_start=101), replace(self.span, source_end=143),
                        replace(self.span, source_start=True), replace(self.span, source_start=-1),
                        replace(self.span, source_end=100), replace(self.span, obligation_id="unknown")):
            with self.subTest(span=changed), self.assertRaises(gr.GroundingError):
                self.bind(scope=replace(self.scope, spans=(changed,)))

    def test_missing_extra_duplicate_or_non_tuple_spans_are_rejected(self):
        for spans in ((), (self.span, self.span), [self.span], ("invalid",)):
            with self.subTest(spans=spans), self.assertRaises(gr.GroundingError):
                self.bind(scope=replace(self.scope, spans=spans))

    def test_wrong_question_mode_or_empty_requirements_are_rejected(self):
        for scope in (replace(self.scope, question="Another question"),
                      replace(self.scope, mode="approved_only"), replace(self.scope, mode="all")):
            with self.subTest(scope=scope), self.assertRaises(gr.GroundingError): self.bind(scope=scope)
        with self.assertRaises(gr.GroundingError): self.bind(obligations=())
        with self.assertRaises(gr.GroundingError):
            self.bind(obligations=(replace(self.ob, validation_mode="evidence_linked"),))

    def test_changed_citation_hash_version_status_or_text_is_rejected(self):
        for ob in (replace(self.ob, source_quote="A reviewer may skip approval."),
                   *(replace(self.ob, citation=replace(self.ob.citation, **fields)) for fields in
                     ({"content_hash": "f" * 64}, {"version": "9"}, {"status": "Approved"},
                      {"section_id": "2"}, {"document_id": "FAKE-001"}, {"chunk_ids": ("fake",)}))):
            with self.subTest(obligation=ob), self.assertRaises(gr.GroundingError): self.bind((ob,))

    def test_ambiguous_exact_and_normalized_occurrences_are_rejected(self):
        p = package("A reviewer must\nreview and approve the plan. " + self.ob.source_quote)
        ob = replace(self.ob, citation=p.groups[0].citation)
        match = gr.match_quotation(ob.source_quote, p.groups[0])
        # Even explicitly selecting one interval cannot silently resolve applicability.
        span = gr.ReviewedExtractiveSpan("review", match.source_start, match.source_end)
        with self.assertRaises(gr.GroundingError):
            self.bind((ob,), replace(self.scope, spans=(span,)), p)

    def test_source_gap_cannot_be_joined_to_form_reviewed_span(self):
        group = self.p.groups[0]
        p = replace(self.p, groups=(replace(group, spans=(
            EvidenceSpan(100, 115, "A reviewer must", ("one",)),
            EvidenceSpan(120, 148, "review and approve the plan.", ("one",)))),))
        with self.assertRaises(gr.GroundingError): self.bind(context=p)

    def test_missing_or_excluded_source_cannot_be_bound(self):
        with self.assertRaises(gr.GroundingError): self.bind(context=replace(self.p, groups=()))


class ExtractiveProfileTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(g, "EXTRACTIVE_SCHEMA_VERSION"), "extractive reference profile is missing")
        self.no_network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)
        self.source = next(c for d in load_documents(ROOT, mode="draft_review") if d.document_id == "PLAT-001"
                           for c in chunk_document(d).chunks if c.section_id == "6.1")
        self.hits = (hit(self.source),)
        self.context = g.prepare_answer(QUESTION, self.hits, mode="draft_review", reference_date=AS_OF).context
        self.obligations = terraform_obligations(self.context)
        self.scope = terraform_scope()
        self.body = refs_payload(self.obligations)

    def prepare(self, **options):
        kwargs = dict(mode="draft_review", reference_date=AS_OF, obligations=self.obligations,
                      extractive_scope=self.scope)
        kwargs.update(options)
        return g.prepare_answer(QUESTION, self.hits, **kwargs)

    def validate(self, body=None, **options):
        kwargs = dict(obligations=self.obligations, extractive_scope=self.scope, question=QUESTION)
        kwargs.update(options)
        body = self.body if body is None else body
        return g.validate_answer(body if isinstance(body, str) else json.dumps(body), self.context, **kwargs)

    def service(self, provider=None, hits=None, backend="chroma"):
        from northstar_atlas.qa import QAService
        provider = FakeLLM(self.body) if provider is None else provider
        return QAService(FixtureStore(self.hits if hits is None else hits), object(), provider,
                         corpus_fingerprint="fixture", backend=backend), provider

    def ask(self, service, **options):
        kwargs = dict(mode="draft_review", reference_date=AS_OF, obligations=self.obligations,
                      extractive_scope=self.scope)
        kwargs.update(options)
        return service.ask(QUESTION, **kwargs)

    def test_complete_four_ids_render_original_source_not_model_wording(self):
        answer = self.validate()
        self.assertEqual(answer.coverage.extractive_ids, tuple(o.obligation_id for o in self.obligations))
        self.assertEqual(answer.coverage.status, "required_entries_present")
        rendered = g.render_answer(answer)
        main = rendered.split("Supporting quotations:")[0]
        source = (ROOT / "data/policies/PLAT-001.md").read_text()
        for reviewed, ob, entry in zip(self.scope.spans, self.obligations, answer.obligation_answers, strict=True):
            exact = source[reviewed.source_start:reviewed.source_end]
            self.assertEqual(entry.statement, exact)
            self.assertIn(exact, main)
            self.assertEqual(entry.statement_origin, "application_reviewed_source")
            self.assertEqual(entry.support[0].quote, ob.source_quote)
            self.assertEqual(entry.support[0].match.kind, "whitespace_normalized")
            self.assertEqual(entry.support[0].citation, ob.citation)
        self.assertIn("Reviewed policy excerpts", main)
        self.assertIn("PLAT-001 v0.1 §6.1 (proposed)", main)
        self.assertIn("do not prove", rendered)

    def test_schema_and_prompt_select_profile_without_changing_legacy(self):
        prepared = self.prepare()
        self.assertEqual(prepared.schema_version, "grounded-answer-v2-extractive-refs")
        entries = prepared.request.schema["properties"]["main_answer"]["properties"]["obligations"]
        self.assertEqual(entries["required"], [o.obligation_id for o in self.obligations])
        for entry in entries["properties"].values():
            self.assertEqual(set(entry["properties"]), {"action"})
            self.assertFalse(entry["additionalProperties"])
        self.assertEqual(len(prepared.extractive_bindings), 4)
        self.assertIn("include_reviewed", prepared.request.system)
        self.assertIn("untrusted", prepared.request.system)
        self.assertIn("other approval conditions", prepared.request.user)
        self.assertLessEqual(prepared.request.input_token_bound + prepared.request.output_tokens,
                             prepared.request.context_tokens)
        old = self.prepare(extractive_scope=None)
        self.assertEqual(old.schema_version, "grounded-answer-v2")
        self.assertEqual(old.request.system, g.SYSTEM + "\nJSON contract: " + g.canonical_json(old.request.schema))

    def test_missing_duplicate_unknown_and_waived_ids_are_rejected(self):
        for key in self.body["main_answer"]["obligations"]:
            body = refs_payload(self.obligations)
            del body["main_answer"]["obligations"][key]
            with self.subTest(missing=key), self.assertRaises(g.GenerationError): self.validate(body)
        body = refs_payload(self.obligations)
        body["main_answer"]["obligations"]["invented"] = {"action": "include_reviewed"}
        with self.assertRaises(g.GenerationError): self.validate(body)
        raw = json.dumps(self.body)
        raw = raw.replace('"stop_and_amend": {"action": "include_reviewed"}',
                          '"stop_and_amend": {"action": "include_reviewed"}, "stop_and_amend": {"action": "include_reviewed"}')
        with self.assertRaises(g.GenerationError): self.validate(raw)
        for action in ("waived", "not_applicable", "complete", None, True):
            body = refs_payload(self.obligations)
            body["main_answer"]["obligations"]["stop_and_amend"]["action"] = action
            with self.subTest(action=action), self.assertRaises(g.GenerationError): self.validate(body)

    def test_extra_wording_quotes_citations_and_claims_are_rejected(self):
        for field in ("statement", "quote", "citation", "status", "complete"):
            body = refs_payload(self.obligations)
            body["main_answer"]["obligations"]["stop_and_amend"][field] = "SECRET_REJECTED"
            with self.subTest(field=field), self.assertRaises(g.GenerationError): self.validate(body)
        body = refs_payload(self.obligations)
        body["main_answer"]["claims"] = [{"text": "extra", "support": []}]
        with self.assertRaises(g.GenerationError): self.validate(body)

    def test_wire_key_order_does_not_change_reviewed_render_order(self):
        body = refs_payload(self.obligations)
        body["main_answer"]["obligations"] = dict(reversed(tuple(body["main_answer"]["obligations"].items())))
        self.assertEqual(self.validate(body), self.validate())

    def test_wrong_spans_or_citations_fail_before_provider(self):
        service, provider = self.service()
        wrong = replace(self.scope.spans[0], source_end=self.scope.spans[0].source_end - 1)
        invalid_scope = replace(self.scope, spans=(wrong, *self.scope.spans[1:]))
        for options in ({"extractive_scope": invalid_scope},
                        {"obligations": (replace(self.obligations[0], citation=replace(
                            self.obligations[0].citation, content_hash="b" * 64)), *self.obligations[1:])}):
            with self.subTest(options=options):
                result = self.ask(service, **options)
                self.assertFalse(result.accepted)
                self.assertEqual(result.public.code, "context_failed")
        self.assertEqual(len(provider.calls), 0)

    def test_scope_cannot_be_reused_for_other_question_or_mode(self):
        for scope in (replace(self.scope, question="Unrelated question"), replace(self.scope, mode="approved_only")):
            with self.subTest(scope=scope), self.assertRaises(g.GenerationError): self.prepare(extractive_scope=scope)
        with self.assertRaises(g.GenerationError): self.validate(question="Unrelated question")
        with self.assertRaises(g.GenerationError): self.validate(question=None)

    def test_draft_cannot_enter_approved_only_even_with_matching_scope(self):
        with self.assertRaises(g.GenerationError):
            self.prepare(mode="approved_only", extractive_scope=replace(self.scope, mode="approved_only"))

    def test_missing_sibling_fails_and_complete_family_preserves_conditions(self):
        parts = family(self.source, [len(self.source.text) // 2])
        self.hits = (hit(parts[0]),)
        with self.assertRaises(g.GenerationError): self.prepare()
        self.hits = (hit(parts[0], parts[1:]),)
        p = g.prepare_answer(QUESTION, self.hits, mode="draft_review", reference_date=AS_OF)
        reviewed = terraform_obligations(p.context)
        p = self.prepare(obligations=reviewed)
        self.assertEqual(len(p.extractive_bindings), 4)
        self.assertIn(parts[1].chunk_id, p.context.groups[0].sources[0].required_context_ids)
        answer = g.validate_answer(json.dumps(refs_payload(reviewed)), p.context,
                                   obligations=reviewed, extractive_scope=self.scope, question=QUESTION)
        self.assertIn("other approval conditions", g.render_answer(answer))
        self.assertIn("amendment or escalation", g.render_answer(answer))

    def test_prompt_budget_failure_prevents_provider_call(self):
        service, provider = self.service()
        result = self.ask(service, limits=GenerationLimits(1024, 128, 64))
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, "context_failed")
        self.assertEqual(provider.calls, [])

    def test_prepared_binding_tamper_is_rejected_before_provider(self):
        prepared = self.prepare()
        wrong = replace(prepared.extractive_bindings[0], match=replace(
            prepared.extractive_bindings[0].match, source_text="Invented policy."))
        provider = FakeLLM(self.body)
        for changed in (replace(prepared, extractive_bindings=(wrong, *prepared.extractive_bindings[1:])),
                        replace(prepared, schema_version=g.SCHEMA_VERSION),
                        replace(prepared, extractive_scope=None),
                        replace(prepared, request=replace(prepared.request, user=prepared.request.user.replace(
                            QUESTION, "Unrelated question")))):
            with self.subTest(), self.assertRaises(g.GenerationError): g.generate_answer(changed, provider)
        self.assertEqual(provider.calls, [])

    def test_answer_cannot_omit_unresolved_condition_or_publish_partial_set(self):
        for kind in ("answer", "clarify", "abstain"):
            body = refs_payload(self.obligations)
            body["main_answer"]["obligations"]["stop_and_amend"] = {"action": "unresolved"}
            body["kind"] = kind
            body["reason"] = {"answer": "supported", "clarify": "missing_details", "abstain": "insufficient_evidence"}[kind]
            if kind == "clarify":
                body["missing_information"] = [g.EXTRACTIVE_UNRESOLVED_TEXT]
                body["clarification_question"] = g.EXTRACTIVE_CLARIFICATION
            if kind == "abstain": body["missing_information"] = [g.NO_EVIDENCE_TEXT]
            with self.subTest(kind=kind), self.assertRaises(g.GenerationError): self.validate(body)

    def test_all_unresolved_can_clarify_or_abstain_without_partial_guidance(self):
        for kind in ("clarify", "abstain"):
            body = refs_payload(self.obligations)
            body["main_answer"]["obligations"] = {o.obligation_id: {"action": "unresolved"} for o in self.obligations}
            body.update(kind=kind, reason="missing_details" if kind == "clarify" else "insufficient_evidence",
                        missing_information=[g.EXTRACTIVE_UNRESOLVED_TEXT if kind == "clarify" else g.NO_EVIDENCE_TEXT],
                        clarification_question=g.EXTRACTIVE_CLARIFICATION if kind == "clarify" else None)
            answer = self.validate(body)
            self.assertEqual(answer.kind, kind)
            self.assertEqual(len(answer.coverage.missing_ids), 4)
            self.assertFalse(any(a.statement for a in answer.obligation_answers))
            self.assertNotIn("Platform may proceed", g.render_answer(answer))
            body["missing_information"] = ["Keep records for 7 years."]
            with self.assertRaises(g.GenerationError): self.validate(body)

    def test_uncited_side_fields_are_rejected(self):
        for key, value in (("missing_information", ["Execution is always authorized."]),
                           ("clarification_question", "Retention is 7 years, correct?")):
            body = refs_payload(self.obligations); body[key] = value
            with self.subTest(key=key), self.assertRaises(g.GenerationError): self.validate(body)

    def test_mixed_evidence_linked_entry_keeps_existing_provenance_checks(self):
        reviewed = (*self.obligations[:-1], replace(self.obligations[-1], validation_mode="evidence_linked"))
        scope = replace(self.scope, spans=self.scope.spans[:-1])
        body = refs_payload(reviewed)
        body["main_answer"]["obligations"]["stop_and_amend"]["statement"] = "Stop and escalate when approved bounds are exceeded."
        answer = self.validate(body, obligations=reviewed, extractive_scope=scope)
        self.assertEqual(answer.coverage.evidence_linked_ids, ("stop_and_amend",))
        self.assertEqual(answer.obligation_answers[-1].statement_origin, "model")
        self.assertIn("require semantic review", g.render_answer(answer))
        body["main_answer"]["obligations"]["stop_and_amend"]["support"][0]["quote"] = "No approvals needed."
        with self.assertRaises(g.GenerationError): self.validate(body, obligations=reviewed, extractive_scope=scope)

    def test_service_uses_profile_and_records_origins_for_either_backend_contract(self):
        for backend in ("chroma", "pinecone"):
            service, provider = self.service(backend=backend)
            result = self.ask(service)
            self.assertTrue(result.accepted)
            self.assertEqual(result.diagnostics.schema_version, "grounded-answer-v2-extractive-refs")
            self.assertEqual(result.diagnostics.application_owned_ids, tuple(o.obligation_id for o in self.obligations))
            self.assertEqual(result.diagnostics.quotation_matches, ("whitespace_normalized",) * 4)
            self.assertEqual(result.public.citations, (self.obligations[0].citation,))
            self.assertEqual(len(provider.calls), 1)

    def test_mixed_unresolved_set_cannot_publish_partial_or_uncited_guidance(self):
        reviewed = (*self.obligations[:-1], replace(self.obligations[-1], validation_mode="evidence_linked"))
        scope = replace(self.scope, spans=self.scope.spans[:-1])
        body = refs_payload(reviewed)
        body.update(kind="clarify", reason="missing_details",
                    missing_information=[g.EXTRACTIVE_UNRESOLVED_TEXT],
                    clarification_question=g.EXTRACTIVE_CLARIFICATION)
        entries = body["main_answer"]["obligations"]
        for ob in reviewed[:-1]: entries[ob.obligation_id] = {"action": "unresolved"}
        # One resolved paraphrase must not escape through an unresolved extractive set.
        with self.assertRaises(g.GenerationError): self.validate(body, obligations=reviewed, extractive_scope=scope)
        entries["stop_and_amend"] = {"statement": None, "support": [],
                                    "unresolved_reason": g.EXTRACTIVE_UNRESOLVED_TEXT}
        answer = self.validate(body, obligations=reviewed, extractive_scope=scope)
        self.assertEqual(len(answer.coverage.missing_ids), 4)
        self.assertNotIn("Platform may proceed", g.render_answer(answer))
        entries["stop_and_amend"]["unresolved_reason"] = "You can proceed without approval."
        with self.assertRaises(g.GenerationError): self.validate(body, obligations=reviewed, extractive_scope=scope)

    def test_malformed_rejected_content_is_not_published_or_retried(self):
        service, provider = self.service()
        self.assertTrue(self.ask(service).accepted)
        for invalid in ("SECRET_REJECTED", '{"kind":"answer"', json.dumps({"kind": "answer"})):
            provider.body = invalid; before = len(provider.calls)
            result = self.ask(service)
            self.assertFalse(result.accepted)
            self.assertEqual(result.public.code, "response_rejected")
            self.assertEqual(len(provider.calls), before + 1)
            self.assertNotIn("SECRET_REJECTED", json.dumps(asdict(result), default=str))
            self.assertNotIn("Platform", result.public.message)

    def test_finish_length_or_provider_timeout_has_no_source_fallback(self):
        service, provider = self.service()
        with patch.object(provider, "generate", return_value=GenerationResponse(
                json.dumps(self.body), MODEL_ID, "length", None, .01, 1)):
            result = self.ask(service)
            self.assertFalse(result.accepted)
            self.assertEqual(result.public.code, "response_rejected")
        provider.error = TimeoutError("SECRET_REJECTED")
        result = self.ask(service)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, "provider_failed")

    def test_render_refuses_modified_application_owned_statement(self):
        answer = self.validate()
        bad = replace(answer.obligation_answers[0], statement="No approval is needed.")
        with self.assertRaises(g.GenerationError):
            g.render_answer(replace(answer, obligation_answers=(bad, *answer.obligation_answers[1:])))

    def test_legacy_extractive_failure_and_linked_behavior_are_unchanged(self):
        body = payload(self.obligations)
        body["main_answer"]["obligations"]["stop_and_amend"]["statement"] = "Stop and escalate beyond approved bounds."
        with self.assertRaises(g.IncompleteAnswerError):
            g.validate_answer(json.dumps(body), self.context, obligations=self.obligations)
        linked = (*self.obligations[:-1], replace(self.obligations[-1], validation_mode="evidence_linked"))
        answer = g.validate_answer(json.dumps(body), self.context, obligations=linked)
        self.assertEqual(answer.coverage.evidence_linked_ids, ("stop_and_amend",))
        with self.assertRaises(g.GenerationError): self.validate(payload(self.obligations))

    def test_ordinary_hr_retention_and_no_evidence_paths_remain_legacy(self):
        from northstar_atlas.qa import QAService
        hr = hit(chunk(doc="HR-001", status=APPROVED, effective=AS_OF))
        prepared = g.prepare_answer("Who?", [hr], mode="approved_only", reference_date=AS_OF)
        provider = FakeLLM(authored_claims(prepared))
        service = QAService(FixtureStore([hr]), object(), provider, corpus_fingerprint="fixture")
        result = service.ask("Who?", reference_date=AS_OF)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.status, "not_assessed")
        self.assertEqual(result.diagnostics.schema_version, "grounded-answer-v2")
        self.assertEqual(result.diagnostics.application_owned_ids, ())
        source = next(c for d in load_documents(ROOT, mode="draft_review") if d.document_id == "COMP-001"
                      for c in chunk_document(d).chunks if c.section_id == "11.1")
        p = g.prepare_answer("Duration?", [hit(source)], mode="draft_review", reference_date=AS_OF)
        requirements = retention_requirements(p.context)
        provider = FakeLLM(authored_retention(requirements))
        service = QAService(FixtureStore([hit(source)]), object(), provider, corpus_fingerprint="fixture")
        result = service.ask("Duration?", mode="draft_review", reference_date=AS_OF, abstention=requirements)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, "abstain")
        self.assertIn("COMP-001 v0.1 §11.1 (proposed)", result.public.text)
        with self.assertRaises(g.GenerationError):
            g.prepare_answer("Duration?", [hit(source)], mode="draft_review", reference_date=AS_OF,
                             abstention=requirements, extractive_scope=self.scope)
        service = QAService(FixtureStore(), object(), FakeLLM(error=AssertionError("must not call")),
                            corpus_fingerprint="fixture")
        result = service.ask("Who?", reference_date=AS_OF)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, "abstain")
        self.assertEqual(result.diagnostics.provider_calls, 0)
        self.assertEqual(result.public.coverage.status, "not_assessed")


if __name__ == "__main__":
    unittest.main()
