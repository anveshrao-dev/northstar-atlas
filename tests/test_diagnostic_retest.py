"""One-shot runner through actual semantic graph and mocked HTTP, no network."""
from dataclasses import replace
from datetime import date, timedelta
import json
import unittest
from unittest.mock import patch

import httpx
import test_live_graph as wiring_fixture
from budget_fixture import BILLING, SCOPE, Provider, document, request
from test_budget_transition import history
from northstar_atlas import budget_amendment as amendment, diagnostic_retest as runner
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.live_budget import BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH


class DiagnosticRunnerTests(unittest.TestCase):
    def setUp(self):
        self.w = wiring_fixture.LiveGraphTests(methodName='runTest')
        self.w.setUp(); self.addCleanup(self.w.doCleanups)
        self.w.path.write_text(json.dumps(document(calls=8, legacy=history())))
        self.old = WorkflowRequest('old-uncertain', GRAPH, 'Previous', 'draft_review', 'chroma', date(2026, 10, 27))
        with self.assertRaises(TimeoutError):
            with self.w.budget.reserve(self.old) as permit:
                permit.invoke(Provider(TimeoutError()), 'router', 0, request(256, 8192))
        self.history = json.loads(self.w.path.read_bytes())
        self.spec = WorkflowRequest('approved-diagnostic', GRAPH, 'Who?', 'draft_review', 'chroma', date(2026, 10, 27))
        self.plan = amendment.prepare_amendment(self.w.budget, amendment.request_contract(self.w.budget, self.spec),
            confirmed_billing_scope=BILLING, approved_max_attempts=15, approved_max_reserved_usd='0.100000',
            writers_stopped=True, dry_run=False)
        amendment.apply_amendment(self.w.budget, self.plan.plan_sha256,
            approved_plan_sha256=self.plan.plan_sha256, writers_stopped=True)
        self.w.budget = PersistentBudget(self.w.root, scope=SCOPE, billing_scope=BILLING,
            policy=BudgetPolicy(15), before_request=self.w.access)

    def run_once(self, **changes):
        options = dict(plan_sha256=self.plan.plan_sha256,
            approved_run_sha256=runner.run_approval(self.plan.plan_sha256),
            question_sha256=sha(self.spec.question.encode()), allow_live=True, other_writers_stopped=True)
        options.update(changes)
        return runner.run_diagnostic(self.w.budget, self.spec, **options)

    def test_complete_seven_stage_run_and_duplicate_rejection(self):
        self.w.wire(judgments=('uncertain', 'relevant'), semantics=('incomplete', 'supported'))
        result = self.run_once()
        self.assertTrue(result.accepted); self.assertEqual(result.kind, 'answer')
        self.assertEqual(self.w.sent, ['router', 'grader', 'grader', 'generator', 'semantic_checker', 'generator', 'semantic_checker'])
        value = json.loads(self.w.path.read_bytes())
        self.assertEqual(value['workflows'][0], self.history['workflows'][0])
        self.assertEqual(value['legacy'], self.history['legacy'])
        self.assertEqual(self.w.budget.inspect_history(diagnostic_approval=self.plan.plan_sha256).attempts_consumed, 15)
        again = self.run_once()
        self.assertFalse(again.accepted); self.assertEqual(len(self.w.sent), 7)
        self.assertEqual(len(self.w.created), 1); self.assertEqual(len(self.w.closed), 1)

    def test_timeout_preserves_full_reservation_and_sanitized_failure(self):
        self.w.wire(failure=httpx.ReadTimeout('SECRET raw response private prompt'))
        result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        self.assertEqual(self.w.sent, ['router'])
        value = json.loads(self.w.path.read_bytes())
        self.assertEqual(value['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(value['workflows'][0], self.history['workflows'][0])
        diag = json.loads(self.w.path.with_name('provider-failure.json').read_bytes())
        self.assertEqual(diag['category'], 'transport_timeout')
        self.assertTrue(diag['transport_attempted']); self.assertNotIn('SECRET', json.dumps(diag))
        self.assertNotIn('SECRET', repr(result)); self.assertFalse(self.run_once().accepted)
        self.assertEqual(len(self.w.sent), 1)

    def test_http_and_response_envelope_failure(self):
        self.w.wire()
        # Replace only the mock transport handler; the real adapter still validates it.
        with patch('httpx.MockTransport.handle_request', return_value=httpx.Response(401, json={'secret': 'PRIVATE'})):
            result = self.run_once()
        self.assertFalse(result.accepted)
        diag = json.loads(self.w.path.with_name('provider-failure.json').read_bytes())
        self.assertEqual((diag['category'], diag['http_status']), ('http_rejection', 401))
        self.assertNotIn('PRIVATE', json.dumps(diag))

    def test_malformed_envelope_is_uncertain_and_not_retried(self):
        self.w.wire()
        with patch('httpx.MockTransport.handle_request', return_value=httpx.Response(200, json={'bad': 'PRIVATE'})) as send:
            result = self.run_once()
        self.assertFalse(result.accepted); self.assertEqual(send.call_count, 1)
        diag = json.loads(self.w.path.with_name('provider-failure.json').read_bytes())
        self.assertEqual(diag['category'], 'response_envelope_validation')
        self.assertEqual(json.loads(self.w.path.read_bytes())['workflows'][-1]['status'], 'uncertain')

    def test_wrong_authorization_or_question_never_constructs_provider(self):
        self.w.wire()
        for changes in (dict(allow_live=False), dict(other_writers_stopped=False),
                        dict(approved_run_sha256='0'*64), dict(question_sha256='0'*64)):
            with self.subTest(changes=changes): self.assertFalse(self.run_once(**changes).accepted)
        self.assertFalse(self.w.created)
        self.assertEqual(len(json.loads(self.w.path.read_bytes())['workflows']), 1)

    def test_exact_configuration_binding_and_no_silent_switch(self):
        self.w.wire(); original = self.spec
        for spec in (replace(original, invocation_id='different'), replace(original, question='Other?'),
                     replace(original, mode='approved_only'), replace(original, backend='custom'),
                     replace(original, reference_date=original.reference_date+timedelta(days=1))):
            self.spec = spec
            self.assertFalse(self.run_once().accepted)
        self.assertFalse(self.w.created)

    def test_expired_pricing_and_budget_mismatch_no_second_pool(self):
        self.w.wire(); now = self.w.budget.clock()
        self.w.budget.clock = lambda: now + timedelta(hours=2)
        self.assertFalse(self.run_once().accepted)
        self.assertFalse(self.w.created)
        self.assertEqual(len(json.loads(self.w.path.read_bytes())['workflows']), 1)

    def test_existing_live_gate_remains_required(self):
        self.w.wire()
        with patch.dict('os.environ', NORTHSTAR_UI_ENABLE_GRAPH='0'):
            self.assertFalse(self.run_once().accepted)
        self.assertFalse(self.w.created)
        self.assertEqual(len(json.loads(self.w.path.read_bytes())['workflows']), 2)
        # Pre-provider failure still consumes the identity, not a new retry ID.
        self.assertFalse(self.run_once().accepted)
        self.assertFalse(self.w.created)

    def test_private_preview_failure_withholds_no_provider(self):
        self.w.wire(); self.w.access.side_effect = ValueError('private failure')
        self.assertFalse(self.run_once().accepted); self.assertFalse(self.w.created)

    def test_rejected_semantic_result_never_published_or_workflow_retried(self):
        self.w.wire(semantics=('unsupported', 'unsupported'))
        result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        self.assertEqual(len(self.w.sent), 6); self.assertEqual(len(self.w.created), 1)
        self.assertFalse(self.run_once().accepted); self.assertEqual(len(self.w.sent), 6)
