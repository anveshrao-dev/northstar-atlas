"""Offline UI contracts: fake QAService results are not retrieval-quality tests."""
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
import unittest

from northstar_atlas.evidence import CitationReference
from northstar_atlas.grounding import ConditionCoverage
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT
from northstar_atlas.qa import (PublicAnswer, PublicError, PublicPresentation, PublicQuotation,
                                PublicStatement, QADiagnostics, QAFailure, QASuccess)
from northstar_atlas.llm import MODEL_ID

from checkout_fixture import ROOT, checkout
from unittest.mock import patch
import os


def success(kind='answer', status=APPROVED):
    citation = CitationReference('fixture', 'HR-001' if status == APPROVED else 'PLAT-001',
                                 '4.2' if status == APPROVED else '6.1', '0.1', status,
                                 'fixture-hash', ('fixture-chunk',))
    return QASuccess(PublicAnswer(kind, 'Fixture public text', (citation,), ConditionCoverage()),
                     QADiagnostics('complete', 'chroma', 'approved_only', date(2026, 10, 23),
                                   'PRIVATE_DIAGNOSTIC', MODEL_ID))


def structured_success(kind='answer', status=APPROVED):
    """Public presentation fixture only; not a real retrieval/grounding run."""
    original = success(kind, status)
    citation = original.public.citations[0]
    presentation = PublicPresentation(
        statements=(PublicStatement('Proceed only within the approved scope.', (citation,)),),
        quotations=(PublicQuotation('Proceed only within the approved scope.\nRecord the comparison.', citation, 'exact'),),
    )
    return replace(original, public=replace(original.public, text='LEGACY_COMBINED_TEXT', presentation=presentation))


class FakeService:
    def __init__(self, result=None):
        self.result = result or success()
        self.calls = []

    def ask(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return self.result


class UIRuntimeTests(unittest.TestCase):
    def setUp(self):
        from northstar_atlas.ui import runtime
        self.ui = runtime
        self.root = checkout(self)
        environment = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root))
        environment.start()
        self.addCleanup(environment.stop)
        self.service = FakeService()
        self.opened, self.closed = [], []

        @contextmanager
        def opener(root, **kwargs):
            self.opened.append((root, kwargs))
            try:
                yield self.service
            finally:
                self.closed.append(kwargs['backend'])
        self.opener = opener

    def owner(self, live=True, pinecone=False):
        owner = self.ui.ResourceOwner(self.root, live_enabled=live, pinecone_enabled=pinecone,
                                      opener=self.opener)
        self.addCleanup(owner.close)
        return owner

    def request(self, ident='one', backend='chroma', mode='approved_only'):
        return self.ui.Submission(ident, 'When?', mode, backend, date(2026, 10, 23))

    def test_projection_excludes_diagnostics_and_keeps_citation_status(self):
        view = self.ui.project_result(success(), 'approved_only')
        self.assertEqual(view.citations, ('HR-001 v0.1 §4.2 (approved, synthetic)',))
        self.assertNotIn('PRIVATE_DIAGNOSTIC', str(asdict(view)))
        self.assertEqual(view.coverage, 'not_assessed')

    def test_structured_projection_preserves_public_objects_and_legacy_text(self):
        result = structured_success()
        view = self.ui.project_result(result, 'approved_only')
        self.assertEqual(view.presentation, result.public.presentation)
        self.assertEqual(view.sources, result.public.citations)
        self.assertEqual(view.text, 'LEGACY_COMBINED_TEXT')
        self.assertNotIn('PRIVATE_DIAGNOSTIC', str(asdict(view)))

    def test_same_display_label_does_not_merge_distinct_source_identities(self):
        result = structured_success()
        first = result.public.citations[0]
        second = replace(first, content_hash='another-source-hash', chunk_ids=('another-chunk',))
        presentation = replace(result.public.presentation, statements=(
            PublicStatement('First source statement.', (first,)),
            PublicStatement('Second source statement.', (second,))))
        result = replace(result, public=replace(result.public, citations=(first, first, second), presentation=presentation))
        view = self.ui.project_result(result, 'approved_only')
        self.assertEqual(len(view.sources), 2)
        self.assertEqual(len(view.citations), 2)

    def test_inconsistent_structured_source_is_withheld(self):
        result = structured_success()
        missing = replace(result.public.citations[0], document_id='INVENTED')
        presentation = replace(result.public.presentation, quotations=(PublicQuotation('Secret candidate', missing, 'exact'),))
        result = replace(result, public=replace(result.public, presentation=presentation))
        view = self.ui.project_result(result, 'approved_only')
        self.assertEqual(view.kind, 'error')
        self.assertNotIn('Secret candidate', str(asdict(view)))

    def test_draft_labels_and_defensive_mode_filter(self):
        view = self.ui.project_result(success(status=DRAFT), 'draft_review')
        self.assertTrue(view.proposed)
        self.assertIn('(proposed)', view.citations[0])
        self.assertEqual(self.ui.project_result(success(status=DRAFT), 'approved_only').kind, 'error')

    def test_clarification_and_abstention_are_distinct(self):
        for kind in ('clarify', 'abstain'):
            self.assertEqual(self.ui.project_result(success(kind), 'approved_only').kind, kind)

    def test_failure_projection_never_publishes_untrusted_message(self):
        result = QAFailure(PublicError('response_rejected', 'SECRET_REJECTED_CANDIDATE'),
                           success().diagnostics)
        view = self.ui.project_result(result, 'draft_review')
        self.assertEqual(view.kind, 'error')
        self.assertIn('withheld', view.text)
        self.assertNotIn('SECRET', str(asdict(view)))
        self.assertFalse(view.citations)

    def test_both_opt_ins_required_and_no_lazy_setup_on_denial(self):
        for enabled, consent in ((False, True), (True, False), (False, False)):
            owner = self.owner(live=enabled)
            self.assertEqual(owner.submit(self.request(), consent=consent).code, 'opt_in_required')
        self.assertEqual(self.opened, [])

    def test_default_opener_can_be_safely_replaced_for_offline_ui_tests(self):
        from unittest.mock import patch
        with patch.object(self.ui, 'open_qa_service', self.opener):
            owner = self.ui.ResourceOwner(self.root, live_enabled=True)
        self.addCleanup(owner.close)
        # Verify injection before allowing any call: never risk a real opener in this test.
        self.assertIs(owner._opener, self.opener)
        owner.submit(self.request(), consent=True)
        self.assertEqual(len(self.opened), 1)

    def test_service_reused_fixed_configuration_and_no_reviewed_obligations(self):
        owner = self.owner()
        owner.submit(self.request(), consent=True)
        owner.submit(self.request('two'), consent=True)
        self.assertEqual(len(self.opened), 1)
        self.assertEqual(len(self.service.calls), 2)
        kwargs = self.service.calls[0][1]
        self.assertEqual(kwargs['top_k'], 5)
        self.assertEqual(kwargs['limits'].output_tokens, 2048)
        self.assertNotIn('obligations', kwargs)
        self.assertNotIn('abstention', kwargs)
        self.assertEqual(self.opened[0][1], dict(backend='chroma', allow_paid=True, model=MODEL_ID))

    def test_duplicate_submission_does_not_call_service_twice(self):
        owner = self.owner()
        owner.submit(self.request(), consent=True)
        self.assertEqual(owner.submit(self.request(), consent=True).code, 'duplicate')
        self.assertEqual(len(self.service.calls), 1)

    def test_backend_switch_is_explicit_and_closes_previous_resource(self):
        owner = self.owner(pinecone=True)
        self.assertFalse(self.opened)
        owner.submit(self.request(), consent=True)
        owner.submit(self.request('two', backend='pinecone'), consent=True)
        self.assertEqual(self.closed, ['chroma'])
        owner.close()
        owner.close()
        self.assertEqual(self.closed, ['chroma', 'pinecone'])
        self.assertEqual(owner.attempts, 2)

    def test_pinecone_disabled_has_no_fallback_or_setup(self):
        view = self.owner().submit(self.request(backend='pinecone'), consent=True)
        self.assertEqual(view.code, 'backend_disabled')
        self.assertFalse(self.opened)

    def test_attempt_cap_survives_disconnect_and_counts_failures(self):
        owner = self.owner()
        self.service.result = QAFailure(PublicError('provider_failed', 'private'), success().diagnostics)
        for n in range(5):
            owner.submit(self.request(str(n)), consent=True)
            owner.close()
        self.assertEqual(owner.submit(self.request('six'), consent=True).code, 'budget_exhausted')
        self.assertEqual(len(self.service.calls), 5)

    def test_busy_owner_does_not_queue_or_consume_attempt(self):
        owner = self.owner()
        with owner.lock:
            view = owner.submit(self.request(), consent=True)
        self.assertEqual(view.code, 'busy')
        self.assertEqual(owner.attempts, 0)

    def test_setup_exception_is_safe_and_not_retried(self):
        @contextmanager
        def broken(*args, **kwargs):
            raise RuntimeError('SECRET_SETUP')
            yield
        owner = self.ui.ResourceOwner(self.root, live_enabled=True, opener=broken)
        view = owner.submit(self.request(), consent=True)
        self.assertEqual(view.code, 'configuration_failed')
        self.assertNotIn('SECRET', str(asdict(view)))
        self.assertEqual(owner.attempts, 1)

    def test_history_is_display_only_and_clear_does_not_reset_attempts(self):
        owner, conversation = self.owner(), self.ui.Conversation()
        conversation.queue(self.request())
        conversation.run_pending(owner, consent=True)
        conversation.run_pending(owner, consent=True)
        self.assertEqual(len(conversation.history), 1)
        self.assertEqual(len(self.service.calls), 1)
        conversation.clear()
        self.assertFalse(conversation.history)
        self.assertEqual(owner.attempts, 1)

    def test_interrupted_turn_is_not_replayed(self):
        owner, conversation = self.owner(), self.ui.Conversation()
        request = self.request()
        conversation.queue(request)
        conversation.started.add(request.turn_id)
        conversation.run_pending(owner, consent=True)
        self.assertEqual(conversation.history[0].view.code, 'interrupted')
        self.assertFalse(self.service.calls)

    def test_invalid_input_rejected_without_initialization(self):
        from dataclasses import replace
        owner = self.owner()
        for request in (replace(self.request(), question=' '),
                        replace(self.request(), question='a' * 2001),
                        replace(self.request(), mode='bad')):
            self.assertEqual(owner.submit(request, consent=True).code, 'invalid_input')
        self.assertFalse(self.opened)


class DemoTests(unittest.TestCase):
    def test_demo_citation_versions_and_status_match_corpus(self):
        from northstar_atlas.ui.demo import DemoQAService, SCENARIOS
        from northstar_atlas.ingestion.document_loader import load_documents
        sources = {d.document_id: d for d in load_documents(ROOT, mode='draft_review')}
        for scenario in SCENARIOS[:3]:
            result = DemoQAService().ask(scenario.question, mode=scenario.mode)
            for citation in result.public.citations:
                source = sources[citation.document_id]
                self.assertEqual(citation.version, source.version)
                self.assertEqual(citation.status, source.status)

    def test_demo_scenarios_and_approved_only_filter(self):
        from northstar_atlas.ui.demo import DemoQAService, SCENARIOS
        from northstar_atlas.ui.runtime import project_result
        service = DemoQAService()
        expected = ('answer', 'answer', 'abstain', 'error', 'clarify')
        for scenario, kind in zip(SCENARIOS, expected, strict=True):
            view = project_result(service.ask(scenario.question, mode=scenario.mode), scenario.mode)
            self.assertEqual(view.kind, kind)
        blocked = service.ask(SCENARIOS[1].question, mode='approved_only')
        self.assertIsInstance(blocked, QAFailure)
        self.assertNotIn('Terraform', blocked.public.message)


if __name__ == '__main__':
    unittest.main()
