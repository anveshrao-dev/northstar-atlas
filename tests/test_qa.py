"""Offline orchestration contracts, not hosted-model semantic evaluation.

FixtureStore supplies controlled evidence; it does not simulate retrieval quality.
FakeLLM returns authored structured responses, never calls an API.
"""
from contextlib import contextmanager
from dataclasses import dataclass, FrozenInstanceError, asdict, replace
from datetime import date
import json
import socket
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, chunk, family, hit
from test_abstention import authored_retention, no_evidence, retention_requirements
from test_generation_v2 import payload
from test_grounding import terraform_obligations
from northstar_atlas import generation as g
from northstar_atlas.evidence import format_citation
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, load_documents
from northstar_atlas.llm import GenerationLimits, GenerationResponse, MODEL_ID, TokenUsage


class FixtureStore:
    def __init__(self, hits=(), error=None):
        self.hits, self.error, self.calls = tuple(hits), error, []

    def search(self, question, embedder, **kwargs):
        self.calls.append((question, embedder, kwargs))
        if self.error:
            raise self.error
        return self.hits


class FakeLLM:
    def __init__(self, body=None, *, error=None):
        self.body, self.error, self.calls = body, error, []

    def generate(self, request):
        self.calls.append(request)
        if self.error:
            raise self.error
        body = self.body(request) if callable(self.body) else self.body
        raw = body if isinstance(body, str) else json.dumps(body)
        return GenerationResponse(raw, request.model, "stop", TokenUsage(100, 50, 150), .01, 1)


def authored_claims(prepared, kind="answer"):
    """Extract supplied source text for a deterministic contract fixture only."""
    body = payload(kind=kind)
    if kind != "abstain":
        body["main_answer"]["claims"] = [
            {"text": group.spans[0].text.strip(), "support": [
                {"citation": format_citation(group.citation), "quote": group.spans[0].text.strip()}]}
            for group in prepared.context.groups]
    return body


class QAServiceTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas import qa
        self.qa = qa
        self.no_network = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def make(self, hits=None, *, body=None, mode="draft_review", error=None, obligations=(), abstention=None):
        hits = (hit(chunk()),) if hits is None else tuple(hits)
        prepared = g.prepare_answer("Who?", hits, mode=mode, reference_date=AS_OF,
                                    obligations=obligations, abstention=abstention)
        provider = FakeLLM(authored_claims(prepared) if body is None else body, error=error)
        store = FixtureStore(hits)
        service = self.qa.QAService(store, object(), provider, corpus_fingerprint="snapshot", backend="chroma")
        return service, store, provider, prepared

    def ask(self, service, **kwargs):
        return service.ask("Who?", mode="draft_review", reference_date=AS_OF, **kwargs)

    def assert_error(self, result, code):
        self.assertIsInstance(result, self.qa.QAFailure)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, code)
        serialized = json.dumps(asdict(result), default=str)
        self.assertNotIn("SECRET_REJECTED", serialized)
        self.assertNotIn("main_answer", serialized)
        self.assertFalse(hasattr(result.public, "kind"))

    def test_approved_answer_default_mode_and_provenance(self):
        source = chunk(doc="HR-001", status=APPROVED, effective=AS_OF)
        service, store, provider, _ = self.make([hit(source)], mode="approved_only")
        result = service.ask("When?", reference_date=AS_OF)
        self.assertIsInstance(result, self.qa.QASuccess)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, "answer")
        self.assertIn("(approved, synthetic)", result.public.text)
        self.assertEqual(result.public.coverage.status, "not_assessed")
        self.assertEqual(result.public.citations[0].content_hash, source.content_hash)
        self.assertEqual(result.diagnostics.evidence[0].source_path, str(source.source_path))
        self.assertEqual(result.diagnostics.evidence[0].source_spans, ((source.source_start, source.source_end, "source"),))
        self.assertEqual(store.calls[0][2], dict(corpus_fingerprint="snapshot", mode="approved_only", top_k=5, reference_date=AS_OF))
        self.assertEqual(result.diagnostics.provider_calls, 1)
        self.assertEqual(result.diagnostics.usage, TokenUsage(100, 50, 150))
        self.assertEqual(len(provider.calls), 1)
        self.assertNotIn("diagnostics", asdict(result.public))

    def test_draft_cross_document_answer_and_immutable_results(self):
        service, _, _, _ = self.make([hit(chunk()), hit(chunk(doc="SEC-001", key="sec"))])
        result = self.ask(service)
        self.assertEqual({c.document_id for c in result.public.citations}, {"IT-001", "SEC-001"})
        self.assertIn("(proposed)", result.public.text)
        self.assertEqual(result.diagnostics.quotation_matches, ("exact", "exact"))
        with self.assertRaises(FrozenInstanceError):
            result.public.text = "changed"
        with self.assertRaises(FrozenInstanceError):
            result.accepted = False

    def test_clarification_preserves_validated_question(self):
        service, _, provider, prepared = self.make()
        provider.body = authored_claims(prepared, "clarify")
        result = self.ask(service)
        self.assertEqual(result.public.kind, "clarify")
        self.assertIn("Which approved scope applies?", result.public.text)

    def test_no_evidence_abstention_skips_provider(self):
        service, _, provider, _ = self.make([], error=AssertionError("must not call"))
        result = self.ask(service)
        self.assertEqual(result.public.kind, "abstain")
        self.assertEqual(result.public.citations, ())
        self.assertIn("retrieval may be incomplete", result.public.text)
        self.assertEqual(result.diagnostics.provider_calls, 0)
        self.assertIsNone(result.diagnostics.usage)
        self.assertEqual(provider.calls, [])

    def test_provider_no_evidence_abstention_with_irrelevant_hits(self):
        service, _, _, _ = self.make(body=no_evidence())
        self.assertEqual(self.ask(service).public.kind, "abstain")

    def test_approved_filter_removes_draft_and_future_evidence(self):
        sources = [hit(chunk()), hit(chunk(key="future", doc="SEC-001", status=APPROVED, effective=date(2099, 1, 1))),
                   hit(chunk(key="hr", doc="HR-001", status=APPROVED, effective=AS_OF))]
        service, _, provider, _ = self.make(sources, mode="approved_only")
        result = service.ask("Who?", reference_date=AS_OF)
        self.assertEqual([c.document_id for c in result.public.citations], ["HR-001"])
        self.assertIn("ineligible", result.diagnostics.warning_codes)
        self.assertEqual(json.loads(provider.calls[0].user)["evidence"]["proposed_evidence"], [])

    def test_invalid_input_stops_before_retrieval(self):
        service, store, provider, _ = self.make()
        for question, options in (("", {}), (None, {}), ("x" * 16001, {}), ("Who?", {"mode": "all"}),
                                  ("Who?", {"top_k": True}), ("Who?", {"top_k": 101}),
                                  ("Who?", {"reference_date": "today"}), ("Who?", {"limits": None})):
            with self.subTest(options=options):
                self.assert_error(service.ask(question, **options), "invalid_input")
        self.assertEqual(store.calls, [])
        self.assertEqual(provider.calls, [])

    def test_missing_sibling_fails_before_provider(self):
        parts = family(chunk("First condition.\nSecond condition.\n"), [17])
        service, store, provider, _ = self.make()
        store.hits = (hit(parts[0]),)
        self.assert_error(self.ask(service), "context_failed")
        self.assertEqual(provider.calls, [])

    def test_complete_siblings_and_trace_are_preserved(self):
        parts = family(chunk("First condition.\nSecond condition.\n"), [17])
        service, _, provider, _ = self.make([hit(parts[0], parts[1:])])
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics.evidence[0].required_context_ids, (parts[1].chunk_id,))
        self.assertIn("Second condition", provider.calls[0].user)
        self.assertEqual(len(result.diagnostics.evidence), 2)

    def test_budget_failure_has_no_provider_call(self):
        service, _, provider, _ = self.make()
        result = self.ask(service, limits=GenerationLimits(1024, 128, 64))
        self.assert_error(result, "context_failed")
        self.assertEqual(provider.calls, [])

    def test_timeout_and_arbitrary_provider_error_are_not_abstentions(self):
        for error in (TimeoutError("SECRET_REJECTED"), RuntimeError("SECRET_REJECTED")):
            service, _, provider, _ = self.make(error=error)
            result = self.ask(service)
            self.assert_error(result, "provider_failed")
            self.assertEqual(result.diagnostics.provider_calls, 1)
            self.assertEqual(len(provider.calls), 1)

    def test_retrieval_failure_is_safe_and_does_not_call_provider(self):
        service, store, provider, _ = self.make()
        store.error = RuntimeError("SECRET_REJECTED")
        self.assert_error(self.ask(service), "retrieval_failed")
        self.assertEqual(provider.calls, [])

    def test_usage_diagnostics_copy_only_allowlisted_fields_on_success_and_rejection(self):
        @dataclass(frozen=True)
        class ExtendedUsage(TokenUsage):
            raw_response: str = "SECRET_REJECTED"

        service, _, provider, prepared = self.make()
        for body in (json.dumps(authored_claims(prepared)), "invalid response"):
            response = GenerationResponse(body, MODEL_ID, "stop", ExtendedUsage(100, 50, 150), .01, 1)
            with patch.object(provider, "generate", return_value=response):
                result = self.ask(service)
            self.assertIs(type(result.diagnostics.usage), TokenUsage)
            self.assertEqual(asdict(result.diagnostics.usage), dict(prompt_tokens=100, completion_tokens=50, total_tokens=150))
            self.assertNotIn("SECRET_REJECTED", json.dumps(asdict(result), default=str))

    def test_malformed_or_uncited_response_is_never_published_or_retried(self):
        service, _, provider, prepared = self.make()
        body = authored_claims(prepared)
        body["main_answer"]["claims"][0]["support"][0]["citation"] = "SECRET_REJECTED"
        for value in (body, "SECRET_REJECTED", {"kind": "answer"}):
            provider.body = value
            before = len(provider.calls)
            result = self.ask(service)
            self.assert_error(result, "response_rejected")
            self.assertEqual(len(provider.calls), before + 1)
            self.assertEqual(result.diagnostics.usage, TokenUsage(100, 50, 150))

    def test_service_reuse_does_not_leak_previous_response(self):
        service, store, provider, _ = self.make()
        accepted = self.ask(service)
        provider.body = "SECRET_REJECTED"
        rejected = self.ask(service)
        self.assert_error(rejected, "response_rejected")
        self.assertNotIn(accepted.public.text, json.dumps(asdict(rejected), default=str))
        provider.body = no_evidence()
        self.assertEqual(self.ask(service).public.kind, "abstain")
        self.assertEqual((len(store.calls), len(provider.calls)), (3, 3))

    def test_terraform_reviewed_obligations_and_rejection(self):
        source = next(c for d in load_documents(ROOT, mode="draft_review") if d.document_id == "PLAT-001"
                      for c in chunk_document(d).chunks if c.section_id == "6.1")
        service, store, provider, prepared = self.make([hit(source)])
        requirements = terraform_obligations(prepared.context)
        provider.body = payload(requirements)
        result = self.ask(service, obligations=requirements)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.public.coverage.extractive_ids), 4)
        for item in requirements:
            self.assertIn(item.source_quote, result.public.text.split("Supporting quotations:")[0])
        # Valid quotes cannot rescue an incomplete main statement.
        provider.body["main_answer"]["obligations"][requirements[0].obligation_id]["statement"] = "SECRET_REJECTED incomplete main statement"
        self.assert_error(self.ask(service, obligations=requirements), "response_rejected")
        unsupported = (replace(requirements[0], source_quote="Invented unrelated obligation."),)
        calls = len(provider.calls)
        self.assert_error(self.ask(service, obligations=unsupported), "context_failed")
        store.hits = ()
        self.assert_error(self.ask(service, obligations=requirements), "context_failed")
        self.assertEqual(len(provider.calls), calls)

    def test_reviewed_retention_abstention_with_valid_citation_and_quote(self):
        source = next(c for d in load_documents(ROOT, mode="draft_review") if d.document_id == "COMP-001"
                      for c in chunk_document(d).chunks if c.section_id == "11.1")
        service, _, provider, prepared = self.make([hit(source)])
        requirements = retention_requirements(prepared.context)
        provider.body = authored_retention(requirements)
        result = self.ask(service, abstention=requirements)
        self.assertEqual(result.public.kind, "abstain")
        self.assertIn("COMP-001 v0.1 §11.1 (proposed)", result.public.text)
        self.assertIn("Documented gap", result.public.text)
        self.assertEqual(result.public.coverage.evidence_linked_ids, ("no_period_established",))
        provider.body = no_evidence()
        self.assert_error(self.ask(service, abstention=requirements), "response_rejected")


class QAOpeningTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas import qa
        self.qa = qa
        self.entered, self.closed = [], []
        self.store = FixtureStore()

        @contextmanager
        def retriever(root, backend):
            self.entered.append((root, backend))
            try:
                yield self.store, object(), "snapshot"
            finally:
                self.closed.append(backend)

        self.retriever = retriever

    def test_default_environment_and_explicit_selection_reuse_one_session(self):
        from northstar_atlas import nebius
        for environment, explicit, expected in (({}, None, "chroma"), ({"VECTOR_STORE": "pinecone"}, None, "pinecone"),
                                                 ({"VECTOR_STORE": "pinecone"}, "chroma", "chroma")):
            with self.subTest(expected=expected), patch.dict("os.environ", environment, clear=True), \
                 patch.object(self.qa, "open_retriever", self.retriever), \
                 patch.object(nebius, "load_settings", side_effect=AssertionError("must not read credentials")):
                with self.qa.open_qa_service(ROOT, backend=explicit, provider=FakeLLM()) as service:
                    first = service.ask("First?")
                    second = service.ask("Second?")
                    self.assertEqual(first.public.kind, "abstain")
                    self.assertEqual(second.diagnostics.backend, expected)
                self.assertEqual(self.entered[-1], (ROOT, expected))
                self.assertEqual(self.closed[-1], expected)
        self.assertEqual(len(self.entered), 3)

    def test_unknown_backend_or_wrong_root_stops_before_io(self):
        with patch.object(self.qa, "open_retriever", side_effect=AssertionError("must not open")):
            for root, backend in ((ROOT, "SECRET_REJECTED"), (ROOT.parent, "chroma"), (ROOT, "")):
                with self.subTest(backend=backend), self.assertRaises(self.qa.QAConfigurationError) as caught:
                    with self.qa.open_qa_service(root, backend=backend, provider=FakeLLM()):
                        self.fail("opened invalid configuration")
                self.assertNotIn("SECRET_REJECTED", str(caught.exception))

    def test_nebius_requires_explicit_opt_in_before_credentials_or_retrieval(self):
        from northstar_atlas import nebius
        with patch.object(nebius, "load_settings", side_effect=AssertionError("credential IO")), \
             patch.object(self.qa, "open_retriever", side_effect=AssertionError("retrieval IO")):
            with self.assertRaises(self.qa.QAConfigurationError):
                with self.qa.open_qa_service(ROOT):
                    self.fail("opened paid configuration")

    def test_invalid_injected_provider_is_rejected_before_backend_or_credential_io(self):
        from northstar_atlas import nebius
        with patch.object(nebius, "load_settings") as credentials, \
             patch.object(self.qa, "open_retriever", self.retriever):
            with self.assertRaises(self.qa.QAConfigurationError):
                with self.qa.open_qa_service(ROOT, backend="pinecone", provider=object()):
                    self.fail("invalid injected provider")
        credentials.assert_not_called()
        self.assertEqual(self.entered, [])

    def test_missing_credentials_is_safe_setup_error_and_no_backend_io(self):
        from northstar_atlas import nebius
        with patch.object(nebius, "load_settings", side_effect=nebius.ProviderError("SECRET_REJECTED")), \
             patch.object(self.qa, "open_retriever", side_effect=AssertionError("retrieval IO")):
            with self.assertRaises(self.qa.QAConfigurationError) as caught:
                with self.qa.open_qa_service(ROOT, allow_paid=True):
                    self.fail("missing credential")
        self.assertEqual(caught.exception.public.code, "configuration_failed")
        self.assertNotIn("SECRET_REJECTED", str(caught.exception))

    def test_configured_nebius_uses_one_attempt_and_closes(self):
        from northstar_atlas import nebius
        settings = nebius.NebiusSettings("fixture-only-not-a-real-key")
        with patch.object(nebius, "load_settings", return_value=settings), \
             patch.object(nebius, "NebiusProvider") as provider, \
             patch.object(self.qa, "open_retriever", self.retriever):
            provider.return_value.__enter__.return_value = FakeLLM()
            with self.qa.open_qa_service(ROOT, allow_paid=True) as service:
                self.assertEqual(service.ask("Who?").public.kind, "abstain")
            provider.assert_called_once_with(settings, allow_paid=True, max_attempts=1)
            provider.return_value.__exit__.assert_called_once()

    def test_setup_failure_is_sanitized_and_closes_owned_resources(self):
        from northstar_atlas import nebius
        settings = nebius.NebiusSettings("fixture-only-not-a-real-key")
        with patch.object(nebius, "load_settings", return_value=settings), \
             patch.object(nebius, "NebiusProvider") as provider, \
             patch.object(self.qa, "open_retriever", side_effect=RuntimeError("SECRET_REJECTED")):
            with self.assertRaises(self.qa.QAConfigurationError) as caught:
                with self.qa.open_qa_service(ROOT, allow_paid=True):
                    self.fail("invalid backend")
            self.assertNotIn("SECRET_REJECTED", str(caught.exception))
            provider.return_value.__exit__.assert_called_once()

    def test_caller_exception_is_not_reclassified_as_setup_failure(self):
        original = RuntimeError("caller failure")
        with patch.object(self.qa, "open_retriever", self.retriever):
            with self.assertRaises(RuntimeError) as caught:
                with self.qa.open_qa_service(ROOT, provider=FakeLLM()):
                    raise original
        self.assertIs(caught.exception, original)
        self.assertEqual(self.closed, ["chroma"])

    def test_cli_preserves_shared_retriever_entry_point(self):
        from northstar_atlas import answers
        self.assertIs(answers.open_retriever, self.qa.open_retriever)


class DatasetCompatibilityTests(unittest.TestCase):
    def test_all_fifteen_questions_with_fixture_evidence_and_authored_fake_responses(self):
        """Ground-truth-selected fixtures exercise plumbing, never retrieval quality."""
        from northstar_atlas.qa import QAService
        dataset = json.loads((ROOT / "evaluations/questions.json").read_text())
        chunks = tuple(c for d in load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
                       for c in chunk_document(d).chunks)
        seen = []
        with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
            for question in dataset["questions"]:
                with self.subTest(question_id=question["question_id"]):
                    selected = tuple(c for c in chunks if any(c.document_id == ref["document_id"] and
                                     (c.section_id == ref["section_id"] or c.section_id.startswith(ref["section_id"] + "."))
                                     for ref in question["required_evidence"]))
                    self.assertTrue(selected)
                    # Broad controlled evidence and budget: NOT a top-five ranking evaluation.
                    limits = GenerationLimits(context_tokens=131072)
                    hits = tuple(hit(c) for c in selected)
                    prepared = g.prepare_answer(question["question"], hits, mode=question["retrieval_mode"],
                                                reference_date=AS_OF, limits=limits)
                    body = authored_claims(prepared, question["expected_behavior"])
                    provider, store = FakeLLM(body), FixtureStore(hits)
                    service = QAService(store, object(), provider, corpus_fingerprint="verified-source-fixture")
                    result = service.ask(question["question"], mode=question["retrieval_mode"],
                                         reference_date=AS_OF, limits=limits, top_k=len(hits))
                    self.assertTrue(result.accepted, result.public)
                    self.assertEqual(result.public.kind, question["expected_behavior"])
                    self.assertEqual(result.public.coverage.status, "not_assessed")
                    self.assertEqual(store.calls[0][0], question["question"])
                    self.assertEqual(result.diagnostics.mode, question["retrieval_mode"])
                    self.assertEqual(len(provider.calls), 1)
                    self.assertTrue(result.diagnostics.evidence)
                    if question["retrieval_mode"] == "approved_only":
                        self.assertTrue(all(t.citation.status == APPROVED for t in result.diagnostics.evidence))
                    seen.append(question["question_id"])
        self.assertEqual(len(set(seen)), 15)


if __name__ == "__main__":
    unittest.main()
