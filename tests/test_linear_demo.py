"""Single Pinecone demonstration using synthetic history and mocked adapters only."""
from contextlib import contextmanager, nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
import io
import multiprocessing
import os
import unittest
from unittest.mock import Mock, patch

import httpx

from budget_fixture import BILLING, SCOPE, Provider, document, request
import test_budget_amendment as amendment_fixture
from test_budget_amendment import encode
from test_context import chunk, hit
from test_qa import FixtureStore, authored_claims
from northstar_atlas import linear_demo as demo
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.generation import prepare_answer
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, LINEAR, GRAPH
from northstar_atlas.llm import MODEL_ID
from northstar_atlas.nebius import NebiusProvider, NebiusSettings


def child_apply(root, plan, previous, pipe):
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(15))
    try:
        demo.apply_demo(budget, plan, approved_plan_sha256=plan,
                        previous_plan_sha256=previous, writers_stopped=True)
    except BudgetError:
        pipe.send('blocked')
    else:
        pipe.send('applied')


class LinearDemoTests(unittest.TestCase):
    def setUp(self):
        fixture = amendment_fixture.AmendmentTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root, self.path = fixture.root, fixture.path
        previous = fixture.prepare(); fixture.apply(previous)
        self.previous = previous.plan_sha256
        self.budget = fixture.updated()
        with self.assertRaises(TimeoutError):
            with self.budget.reserve(fixture.spec, diagnostic_approval=self.previous) as permit:
                permit.invoke(Provider(TimeoutError()), 'router', 0, request(256, 8192))
        self.original = self.path.read_bytes()
        self.history = json.loads(self.original)
        self.window = self.path.with_name('diagnostic-window.json')
        self.original_window = self.window.read_bytes()
        self.spec = WorkflowRequest('one-linear-demo', LINEAR, 'Who?', 'draft_review',
                                    'pinecone', fixture.spec.reference_date)
        self.contract = demo.request_contract(self.budget, self.spec)
        self.price = document()['pricing_revisions'][0]
        self.price['id'] = 'independently-reviewed-demo-price'

    def prepare(self, **changes):
        options = dict(previous_plan_sha256=self.previous, confirmed_billing_scope=BILLING,
                       approved_max_attempts=16, approved_max_reserved_usd='0.100000',
                       writers_stopped=True, pricing_revision=self.price,
                       verified_pricing_sha256=sha(encode(self.price)), dry_run=False)
        options.update(changes)
        return demo.prepare_demo(self.budget, self.contract, **options)

    def apply(self, preview, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256,
                       previous_plan_sha256=self.previous, writers_stopped=True)
        options.update(changes)
        return demo.apply_demo(self.budget, preview.plan_sha256, **options)

    def updated(self):
        return PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                policy=BudgetPolicy(16), before_request=getattr(self, 'access', None))

    def activate_fixture(self):
        self.preview = self.prepare(); self.apply(self.preview)
        self.access = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.budget = self.updated()
        self.enterContext(patch.dict(os.environ, NORTHSTAR_CONTROLLED_LIVE='1'))

    def run_once(self, **changes):
        options = dict(plan_sha256=self.preview.plan_sha256,
                       approved_run_sha256=demo.run_approval(self.preview.plan_sha256),
                       question_sha256=sha(self.spec.question.encode()),
                       allow_live=True, other_writers_stopped=True)
        options.update(changes)
        return demo.run_demo(self.budget, self.spec, **options)

    def wire(self, failure=None, retrieval_failure=None, *, mock_retriever=True):
        self.sent, self.created = [], []
        self.store = FixtureStore([hit(chunk())], error=retrieval_failure)
        prepared = prepare_answer(self.spec.question, self.store.hits, mode=self.spec.mode,
                                  reference_date=self.spec.reference_date)
        def send(req):
            value = json.loads(self.path.read_bytes())
            self.assertEqual(value['workflows'][-1]['profile'], LINEAR)
            self.assertEqual(value['workflows'][-1]['calls'][-1]['status'], 'started')
            self.sent.append(req)
            if failure: raise failure
            return httpx.Response(200, json={'model': MODEL_ID, 'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps(authored_claims(prepared))}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})
        @contextmanager
        def provider(settings, **kwargs):
            value = json.loads(self.path.read_bytes())
            self.assertEqual((value['workflows'][-1]['profile'], value['workflows'][-1]['status']),
                             (LINEAR, 'reserved'))
            self.assertTrue(kwargs['requires_budget']); self.assertEqual(kwargs['max_attempts'], 1)
            self.created.append(True)
            with NebiusProvider(settings, transport=httpx.MockTransport(send), **kwargs) as p:
                yield p
        @contextmanager
        def retriever(root, backend):
            self.assertEqual(backend, 'pinecone')
            yield self.store, object(), 'fixture-corpus'
        self.enterContext(patch('northstar_atlas.nebius.load_settings', return_value=NebiusSettings('synthetic-only')))
        self.enterContext(patch('northstar_atlas.nebius.NebiusProvider', side_effect=provider))
        if mock_retriever:
            self.enterContext(patch('northstar_atlas.qa.open_retriever', side_effect=retriever))

    def wire_pinecone(self, *, audit_failure=None):
        self.wire(mock_retriever=False)
        self.events = []
        def confirmed(*args): self.events.append('confirmation')
        def audit():
            self.events.append('audit')
            if audit_failure: raise audit_failure
        search = self.store.search
        def audited_search(*args, **kwargs):
            self.assertEqual(self.events, ['confirmation', 'audit'])
            return search(*args, **kwargs)
        self.store.audit = audit
        self.store.search = audited_search
        self.enterContext(patch('northstar_atlas.local_embeddings.MiniLMProvider', return_value=object()))
        self.enterContext(patch('northstar_atlas.vector_store.corpus_snapshot', return_value=(None, 'fixture-corpus')))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_cloud.load_settings', return_value=object()))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_cloud.check_confirmation', side_effect=confirmed))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_cloud.sdk_client', return_value=nullcontext(object())))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_workflow.RequestMeter', return_value=Mock()))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_workflow.connect_target',
                                return_value=(object(), nullcontext())))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_workflow.active_snapshot', return_value=object()))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_workflow.open_store', return_value=self.store))

    def test_dry_run_and_byte_exact_backups(self):
        before = {p.name for p in self.path.parent.iterdir()}
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual({p.name for p in self.path.parent.iterdir()}, before)
        saved = self.prepare()
        self.assertEqual(saved.plan_sha256, preview.plan_sha256)
        self.assertEqual((saved.artifact_directory/'backup.v2.json').read_bytes(), self.original)
        self.assertEqual((saved.artifact_directory/'backup.window.json').read_bytes(), self.original_window)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.window.read_bytes(), self.original_window)

    def test_exact_amendment_preserves_both_uncertain_workflows_and_prices(self):
        before = self.budget.inspect_history(diagnostic_approval=self.previous)
        preview = self.prepare(); self.apply(preview)
        value = json.loads(self.path.read_bytes())
        expected = deepcopy(self.history)
        expected.update(max_attempts=16, diagnostic_request_sha256=sha(encode(self.contract)),
                        active_pricing=self.price['id'])
        expected['pricing_revisions'].append(self.price)
        self.assertEqual(value, expected)
        self.assertEqual([w['status'] for w in value['workflows']], ['uncertain', 'uncertain'])
        self.assertEqual(self.updated().inspect_history(diagnostic_approval=preview.plan_sha256), before)
        with self.assertRaises(BudgetError): self.updated().check_ready()

    def test_wrong_cap_cost_scope_or_missing_approval_rejected(self):
        for changes in (dict(approved_max_attempts=17), dict(approved_max_attempts=15),
                        dict(approved_max_reserved_usd='0.20'), dict(writers_stopped=False),
                        dict(confirmed_billing_scope='other'), dict(previous_plan_sha256='0'*64),
                        dict(verified_pricing_sha256='0'*64)):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.prepare(**changes)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_cross_backend_and_profile_contracts_rejected(self):
        for backend, profile in (('chroma', LINEAR), ('custom', LINEAR), ('chroma', GRAPH), ('pinecone', GRAPH)):
            self.contract = {**demo.request_contract(self.budget, self.spec), 'backend': backend, 'profile': profile}
            with self.subTest(backend=backend, profile=profile), self.assertRaises(BudgetError): self.prepare()

    def test_candidate_and_source_bytes_and_plan_approval_bound(self):
        preview = self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview, approved_plan_sha256='0'*64)
        candidate = preview.artifact_directory/'candidate.v2.json'
        raw = candidate.read_bytes(); candidate.write_bytes(raw+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        candidate.write_bytes(raw); self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.window.read_bytes(), self.original_window)

    def test_stale_price_and_duplicate_price_rejected(self):
        for price in (self.history['pricing_revisions'][0],
                      {**self.price, 'confirmed_at': '2020-01-01T00:00:00+00:00',
                       'valid_until': '2020-01-01T01:00:00+00:00'}):
            with self.assertRaises(BudgetError):
                self.prepare(pricing_revision=price, verified_pricing_sha256=sha(encode(price)))
        preview = self.prepare(); now = self.budget.clock()
        self.budget.clock = lambda: now+timedelta(hours=2)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_missing_window_or_ledger_never_creates_second_pool(self):
        self.window.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.window.exists()); self.assertEqual(self.path.read_bytes(), self.original)
        self.window.write_bytes(self.original_window); self.window.chmod(0o600); self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.path.exists())

    def test_failed_write_retains_guard_and_blocks_every_entrypoint(self):
        preview = self.prepare()
        with patch.object(self.budget, '_write', side_effect=OSError('synthetic persistence failure')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), self.original)
        with self.assertRaises(BudgetError): self.updated().check_ready()
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass

    def test_one_real_qa_service_call_and_no_retry_or_other_submission(self):
        self.activate_fixture(); self.wire()
        result = self.run_once()
        self.assertTrue(result.accepted); self.assertEqual(result.kind, 'answer')
        self.assertEqual(len(self.sent), 1); self.assertEqual(len(self.created), 1)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'][:2], self.history['workflows'])
        self.assertEqual(value['workflows'][-1]['profile'], LINEAR)
        self.assertEqual(self.budget.inspect_history(diagnostic_approval=self.preview.plan_sha256).attempts_consumed, 16)
        self.assertFalse(self.run_once().accepted); self.assertEqual(len(self.sent), 1)
        with self.assertRaises(BudgetError):
            with self.budget.reserve(replace(self.spec, invocation_id='unrelated')): pass

    def test_request_identity_and_explicit_authorization_before_provider(self):
        self.activate_fixture(); self.wire()
        for changes in (dict(allow_live=False), dict(other_writers_stopped=False),
                        dict(approved_run_sha256='0'*64), dict(question_sha256='0'*64)):
            self.assertFalse(self.run_once(**changes).accepted)
        original = self.spec
        for spec in (replace(original, invocation_id='other'), replace(original, question='Other?'),
                     replace(original, mode='approved_only'), replace(original, backend='chroma'),
                     replace(original, profile=GRAPH, backend='chroma'),
                     replace(original, reference_date=original.reference_date+timedelta(days=1))):
            self.spec = spec; self.assertFalse(self.run_once().accepted)
        self.assertFalse(self.created); self.assertFalse(self.sent)
        self.assertEqual(self.path.read_bytes(), (self.preview.artifact_directory/'candidate.v2.json').read_bytes())

    def test_timeout_remains_uncertain_sanitized_and_single_attempt(self):
        self.activate_fixture(); self.wire(failure=httpx.ReadTimeout('PRIVATE'))
        result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'][:2], self.history['workflows'])
        self.assertEqual(value['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(self.budget.inspect_history(diagnostic_approval=self.preview.plan_sha256).attempts_consumed, 16)
        self.assertNotIn('PRIVATE', self.path.with_name('provider-failure.json').read_text())
        self.assertFalse(self.run_once().accepted); self.assertEqual(len(self.sent), 1)

    def test_access_or_remote_failure_no_inference_and_no_second_identity(self):
        self.activate_fixture(); self.wire(retrieval_failure=ValueError('private audit failure'))
        self.access.side_effect = ValueError('private preview unavailable')
        self.assertFalse(self.run_once().accepted); self.assertFalse(self.created)
        self.access.side_effect = None
        self.assertFalse(self.run_once().accepted); self.assertFalse(self.sent)

    def test_remote_setup_failure_prevents_inference(self):
        self.activate_fixture(); self.wire()
        with patch('northstar_atlas.qa.open_retriever', side_effect=ValueError('audit failed')):
            result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public); self.assertFalse(self.sent)
        self.assertFalse(self.run_once().accepted)

    def test_provider_cannot_bypass_reservation(self):
        self.activate_fixture()
        with NebiusProvider(NebiusSettings('synthetic-only'), allow_paid=True, requires_budget=True,
                transport=httpx.MockTransport(lambda _: self.fail('unbudgeted transport'))) as p:
            with self.assertRaises(ValueError): p.generate(request())

    def test_existing_pinecone_confirmation_and_audit_precede_inference(self):
        self.activate_fixture(); self.wire_pinecone()
        result = self.run_once()
        self.assertTrue(result.accepted)
        self.assertEqual(self.events, ['confirmation', 'audit'])
        self.assertEqual(len(self.store.calls), 1); self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.store.calls[0][2]['mode'], 'draft_review')

    def test_remote_audit_failure_withholds_and_never_calls_nebius(self):
        self.activate_fixture(); self.wire_pinecone(audit_failure=ValueError('private remote mismatch'))
        self.assertFalse(self.run_once().accepted)
        self.assertEqual(self.events, ['confirmation', 'audit'])
        self.assertFalse(self.sent); self.assertFalse(self.store.calls)
        self.assertFalse(self.run_once().accepted)

    def test_confirmation_failure_prevents_remote_open_and_inference(self):
        self.activate_fixture(); self.wire_pinecone()
        with patch('northstar_atlas.vector_store.pinecone_cloud.check_confirmation', side_effect=ValueError()), \
                patch('northstar_atlas.vector_store.pinecone_cloud.sdk_client') as client:
            self.assertFalse(self.run_once().accepted)
        client.assert_not_called(); self.assertFalse(self.sent)

    def test_lock_contention_across_processes(self):
        preview = self.prepare(); ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        with self.budget._locked(diagnostic_approval=self.previous):
            process = ctx.Process(target=child_apply, args=(self.root, preview.plan_sha256, self.previous, child))
            process.start(); self.assertTrue(parent.poll(15)); self.assertEqual(parent.recv(), 'blocked')
            process.join(15); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_lock_and_window_identity_changes_invalidate_candidate(self):
        preview = self.prepare()
        self.window.write_bytes(self.original_window+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.window.write_bytes(self.original_window)
        self.budget.lock_path.rename(self.budget.directory/'original-lock')
        self.budget.lock_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_readback_failure_preserves_guard_and_no_automatic_rollback(self):
        preview = self.prepare(); write = self.budget._write
        def uncertain(value):
            write(value)
            raise OSError('synthetic post-replacement failure')
        with patch.object(self.budget, '_write', side_effect=uncertain), self.assertRaises(BudgetError):
            self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(json.loads(self.path.read_bytes())['max_attempts'], 16)
        self.assertEqual((preview.artifact_directory/'backup.v2.json').read_bytes(), self.original)
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_missing_active_window_does_not_reopen_ordinary_linear_or_graph(self):
        self.activate_fixture(); self.window.unlink()
        for request in (self.spec, replace(self.spec, backend='chroma'),
                        replace(self.spec, profile=GRAPH, backend='chroma')):
            with self.assertRaises(BudgetError):
                with self.budget.reserve(request): pass
        with self.assertRaises(BudgetError):
            with self.budget.reserve(self.spec, diagnostic_approval=self.preview.plan_sha256): pass

    def test_fresh_price_cost_must_fit_remaining_ceiling(self):
        price = {**self.price, 'input_per_million_usd': '1000'}
        with self.assertRaises(BudgetError):
            self.prepare(pricing_revision=price, verified_pricing_sha256=sha(encode(price)))
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_cli_dry_run_requires_concrete_approval_without_authoritative_writes(self):
        contract = self.root/'.local/demo-contract.json'; price = self.root/'.local/demo-price.json'
        contract.write_bytes(encode(self.contract)); contract.chmod(0o600)
        price.write_bytes(encode(self.price)); price.chmod(0o600)
        output = io.StringIO()
        with patch.dict(os.environ, NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
                        NORTHSTAR_BUDGET_MAX_ATTEMPTS='15'), redirect_stdout(output):
            code = demo.main(['--root', str(self.root), 'prepare', '--contract', str(contract),
                '--previous-plan', self.previous, '--confirmed-billing-scope', BILLING,
                '--approve-max-attempts', '16', '--approve-max-reserved-usd', '0.100000',
                '--pricing-revision', str(price), '--verified-pricing-sha256', sha(encode(self.price)),
                '--writers-stopped'])
        self.assertEqual(code, 0)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.window.read_bytes(), self.original_window)
        for private in (BILLING, str(self.root), self.spec.question): self.assertNotIn(private, output.getvalue())
