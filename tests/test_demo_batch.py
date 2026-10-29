"""Synthetic batch histories only; no providers or real budget state."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import json
import multiprocessing
import os
import unittest
from unittest.mock import patch

from budget_fixture import SCOPE, Provider, request
import test_demo_budget_amendment as fixture_module
from test_demo_budget_amendment import BILLING, encode
from northstar_atlas import demo_batch as batch
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, GRAPH
from northstar_atlas.diagnostic_window import sha


def child_reserve(root, plan, spec, pipe):
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(7))
    try:
        with budget.reserve(spec, diagnostic_approval=plan) as permit:
            permit.invoke(Provider(), 'generator', 0, request())
    except BudgetError:
        pipe.send('denied')
    else:
        pipe.send('reserved')


class BatchTests(unittest.TestCase):
    def setUp(self):
        fixture = fixture_module.DemoAmendmentTests('runTest'); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        previous = fixture.prepare(); fixture.apply(previous)
        self.root, self.path = fixture.root, fixture.path
        self.budget = fixture.updated()
        self.spec = replace(fixture.spec, invocation_id='batch-call', question='An arbitrary policy question?')
        with self.budget.reserve(fixture.spec, diagnostic_approval=previous.plan_sha256) as permit:
            permit.invoke(Provider(), 'generator', 0, request())
        self.previous = previous.plan_sha256
        self.original = self.path.read_bytes(); self.history = json.loads(self.original)
        self.contract = batch.batch_contract(self.budget, 'first-batch')
        self.price = self.history['pricing_revisions'][0]

    def prepare(self, **kw):
        options = dict(previous_plan_sha256=self.previous, confirmed_billing_scope=BILLING,
            approved_max_attempts=7, approved_max_reserved_usd='0.100000', writers_stopped=True,
            verified_pricing_sha256=sha(encode(self.price)), dry_run=False)
        options.update(kw)
        return batch.prepare_batch(self.budget, self.contract, **options)

    def apply(self, preview, **kw):
        options = dict(previous_plan_sha256=self.previous, approved_plan_sha256=preview.plan_sha256,
                       writers_stopped=True)
        options.update(kw)
        return batch.apply_batch(self.budget, preview.plan_sha256, **options)

    def activate(self):
        preview = self.prepare(); self.apply(preview)
        budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(7))
        return preview, budget

    def invoke(self, budget, plan, index, failure=None):
        spec = replace(self.spec, invocation_id=f'question-{index}',
                       mode='draft_review' if index % 2 else 'approved_only')
        with budget.reserve(spec, diagnostic_approval=plan) as permit:
            permit.invoke(Provider(failure), 'generator', 0, request())

    def test_history_backup_dry_run_and_exact_candidate(self):
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(self.path.read_bytes(), self.original)
        saved = self.prepare(); self.assertEqual(saved.plan_sha256, preview.plan_sha256)
        self.assertEqual((saved.artifact_directory/'backup.v2.json').read_bytes(), self.original)
        self.apply(saved)
        current = json.loads(self.path.read_bytes())
        expected = deepcopy(self.history)
        expected.update(max_attempts=7, diagnostic_request_sha256=sha(encode(self.contract)))
        self.assertEqual(current, expected)
        self.assertEqual(Decimal(saved.reserved_usd), Decimal('0.007374'))

    def test_five_questions_restart_duplicates_and_exhaustion(self):
        preview, budget = self.activate()
        for i in range(5):
            self.assertEqual(batch.remaining(budget, preview.plan_sha256), 5-i)
            self.invoke(budget, preview.plan_sha256, i)
            with self.assertRaises(BudgetError): self.invoke(budget, preview.plan_sha256, i)
            budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(7))
        self.assertEqual(batch.remaining(budget, preview.plan_sha256), 0)
        with self.assertRaises(BudgetError): self.invoke(budget, preview.plan_sha256, 5)
        current = json.loads(self.path.read_bytes())
        self.assertEqual(current['workflows'][:2], self.history['workflows'])
        self.assertEqual(current['pricing_revisions'], self.history['pricing_revisions'])
        self.assertEqual(len(current['workflows']), 7)

    def test_zero_call_failure_does_not_reopen_submission(self):
        preview, budget = self.activate()
        for i in range(5):
            with budget.reserve(replace(self.spec, invocation_id=f'empty-{i}'),
                                diagnostic_approval=preview.plan_sha256): pass
        self.assertEqual(batch.remaining(budget, preview.plan_sha256), 0)
        with self.assertRaises(BudgetError): self.invoke(budget, preview.plan_sha256, 8)

    def test_future_batch_requires_distinct_explicit_amendment(self):
        preview, budget = self.activate()
        self.budget = budget; self.previous = preview.plan_sha256
        self.contract = batch.batch_contract(budget, 'second-batch')
        with self.assertRaises(BudgetError): self.prepare(approved_max_attempts=12)
        for i in range(5): self.invoke(budget, preview.plan_sha256, i)
        source = self.path.read_bytes()
        future = self.prepare(approved_max_attempts=12); self.apply(future)
        updated = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(12))
        self.assertEqual(batch.remaining(updated, future.plan_sha256), 5)
        self.assertEqual(json.loads(source)['workflows'], json.loads(self.path.read_bytes())['workflows'])
        with self.assertRaises(BudgetError): batch.remaining(updated, preview.plan_sha256)

    def test_wrong_scope_profile_backend_and_corpus(self):
        preview, budget = self.activate()
        for spec in (replace(self.spec, profile=GRAPH), replace(self.spec, backend='chroma')):
            with self.assertRaises(BudgetError):
                with budget.reserve(spec, diagnostic_approval=preview.plan_sha256): self.fail('unauthorized')
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec): self.fail('unrestricted entrypoint')
        manifest = self.root/'data/corpus_manifest.json'
        manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaises(BudgetError): batch.remaining(budget, preview.plan_sha256)

    def test_approvals_source_and_candidate_binding(self):
        for changes in (dict(writers_stopped=False), dict(approved_max_attempts=8),
                        dict(approved_max_reserved_usd='0.20'), dict(confirmed_billing_scope='other'),
                        dict(verified_pricing_sha256='0'*64)):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.prepare(**changes)
        preview = self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview, approved_plan_sha256='0'*64)
        self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.path.write_bytes(self.original)
        candidate = preview.artifact_directory/'candidate.v2.json'
        candidate.write_bytes(candidate.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_expiry_and_failed_persistence_fail_closed(self):
        with patch.object(self.budget, 'clock', return_value=self.budget.clock()+timedelta(days=2)):
            with self.assertRaises(BudgetError): self.prepare()
        preview = self.prepare()
        with patch.object(self.budget, '_write', side_effect=OSError('synthetic')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_uncertain_reservation_and_history_tamper(self):
        preview, budget = self.activate()
        with self.assertRaises(TimeoutError): self.invoke(budget, preview.plan_sha256, 0, TimeoutError())
        current = json.loads(self.path.read_bytes())
        self.assertEqual(current['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(batch.remaining(budget, preview.plan_sha256), 4)
        current['workflows'][0]['id'] = 'changed-history'
        self.path.write_bytes(encode(current))
        with self.assertRaises(BudgetError): batch.remaining(budget, preview.plan_sha256)

    def enable_runtime(self, failure=None):
        from northstar_atlas.ui.runtime import ResourceOwner
        from test_context import chunk
        from northstar_atlas.ingestion.document_loader import APPROVED
        import test_linear_demo
        preview, budget = self.activate()
        access = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        budget.before_request = access
        self.budget = budget
        self.enterContext(patch.dict(os.environ, NORTHSTAR_CONTROLLED_LIVE='1',
            NORTHSTAR_DEMO_BATCH_ENABLED='1', NORTHSTAR_DEMO_BATCH_PLAN=preview.plan_sha256,
            NORTHSTAR_DEMO_BATCH_RUN_APPROVAL=batch.run_approval(preview.plan_sha256),
            NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1', NORTHSTAR_UI_ENABLE_GRAPH='0'))
        self.enterContext(patch('test_linear_demo.chunk', return_value=chunk(status=APPROVED,
                                                                          effective=self.spec.reference_date)))
        test_linear_demo.LinearDemoTests.wire(self, failure=failure)
        owner = ResourceOwner(self.root, live_enabled=True, pinecone_enabled=True, budget=budget)
        self.addCleanup(owner.close)
        return preview, owner

    def submission(self, turn='user-1', **kw):
        from northstar_atlas.ui.runtime import Submission
        return replace(Submission(turn, self.spec.question, self.spec.mode, 'pinecone',
                                  self.spec.reference_date), **kw)

    def test_real_qa_path_five_typed_answers_recorded_before_projection(self):
        from northstar_atlas.ui import runtime
        preview, owner = self.enable_runtime()
        original_project = runtime.project_result
        def project(result, mode):
            files = list((self.root/'.local/linear-demo-results').glob('*.json'))
            self.assertEqual(len(files), len(self.sent))
            self.assertEqual(json.loads(self.path.read_bytes())['workflows'][-1]['status'], 'closed')
            self.assertTrue(all(b'in_progress' not in p.read_bytes() for p in files))
            return original_project(result, mode)
        with patch.object(runtime, 'project_result', side_effect=project):
            for i in range(5):
                result = owner.submit(self.submission(f'user-{i}', question=f'Question {i}?',
                    mode='draft_review' if i % 2 else 'approved_only'), consent=True)
                self.assertEqual(result.kind, 'answer', result)
                self.assertTrue(result.citations)
                self.assertTrue(result.presentation.statements)
        self.assertEqual(owner.batch_remaining(), 0)
        self.assertEqual(len(self.sent), 5)
        self.assertEqual(len(self.created), 5)
        self.assertEqual(owner.submit(self.submission('sixth'), consent=True).code, 'batch_exhausted')
        self.assertEqual(len(self.sent), 5)

    def test_duplicate_and_formatter_failure_never_regenerate(self):
        from northstar_atlas.ui.runtime import ResourceOwner
        _, owner = self.enable_runtime()
        with patch('northstar_atlas.ui.runtime.project_result', side_effect=RuntimeError('private fixture')):
            result = owner.submit(self.submission(), consent=True)
        self.assertEqual(result.code, 'internal_error')
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(owner.submit(self.submission(), consent=True).code, 'duplicate')
        other = ResourceOwner(self.root, live_enabled=True, pinecone_enabled=True, budget=self.budget)
        self.addCleanup(other.close)
        self.assertEqual(other.submit(self.submission(), consent=True).kind, 'error')
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(owner.batch_remaining(), 4)
        raw = (self.root/'.local/linear-demo-results/user-1.json').read_bytes()
        self.assertNotIn(b'private fixture', raw)
        self.assertNotIn(b'in_progress', raw)

    def test_timeout_retains_full_reservation_and_safe_failure_capture(self):
        import httpx
        _, owner = self.enable_runtime(failure=httpx.ReadTimeout('PRIVATE synthetic payload'))
        result = owner.submit(self.submission(), consent=True)
        self.assertEqual(result.kind, 'error')
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(owner.batch_remaining(), 4)
        raw = (self.root/'.local/linear-demo-results/user-1.json').read_bytes()
        self.assertNotIn(b'PRIVATE', raw)
        self.assertNotIn(b'synthetic-only', raw)

    def test_activation_access_and_backend_checks_precede_provider(self):
        from northstar_atlas.ui.runtime import LANGGRAPH_WORKFLOW
        _, owner = self.enable_runtime()
        with patch.dict(os.environ, NORTHSTAR_DEMO_BATCH_RUN_APPROVAL='0'*64):
            self.assertEqual(owner.submit(self.submission(), consent=True).code, 'budget_unavailable')
        self.assertEqual(owner.submit(self.submission(backend='chroma'), consent=True).code, 'backend_disabled')
        self.assertEqual(owner.submit(self.submission(workflow=LANGGRAPH_WORKFLOW), consent=True).code, 'workflow_disabled')
        self.assertEqual(owner.submit(self.submission(), consent=False).code, 'opt_in_required')
        self.assertEqual(len(self.sent), 0)
        with patch('northstar_atlas.live_preflight.require_private_preview', side_effect=ValueError('private')) as denied:
            self.budget.before_request = denied
            self.assertEqual(owner.submit(self.submission(), consent=True).kind, 'error')
        self.assertEqual(len(self.created), 0)
        self.assertEqual(owner.batch_remaining(), 4)

    def test_settlement_failure_withholds_accepted_answer(self):
        _, owner = self.enable_runtime()
        with patch.object(self.budget, '_settle', side_effect=BudgetError('uncertain')):
            result = owner.submit(self.submission(), consent=True)
        self.assertEqual(result.kind, 'error')
        self.assertEqual(len(self.sent), 1)
        raw = (self.root/'.local/linear-demo-results/user-1.json').read_bytes()
        self.assertNotIn(b'accepted', raw)
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][-1]['status'], 'reserved')

    def test_filesystem_identity_and_price_tamper(self):
        preview = self.prepare()
        self.budget.lock_path.rename(self.path.parent/'retired.lock')
        self.budget.lock_path.touch(mode=0o600)
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(2))
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_cross_process_lock_and_duplicate_serialization(self):
        preview, budget = self.activate()
        ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        with budget._locked(diagnostic_approval=preview.plan_sha256):
            proc = ctx.Process(target=child_reserve, args=(self.root, preview.plan_sha256, self.spec, child))
            proc.start(); self.assertTrue(parent.poll(20)); self.assertEqual(parent.recv(), 'denied')
            proc.join(20); self.assertEqual(proc.exitcode, 0)
        procs = [ctx.Process(target=child_reserve, args=(self.root, preview.plan_sha256, self.spec, child))
                 for _ in range(2)]
        for proc in procs: proc.start()
        results = []
        for _ in procs:
            self.assertTrue(parent.poll(20)); results.append(parent.recv())
        for proc in procs: proc.join(20); self.assertEqual(proc.exitcode, 0)
        self.assertEqual(sorted(results), ['denied', 'reserved'])
        self.assertEqual(batch.remaining(budget, preview.plan_sha256), 4)
        parent.close(); child.close()

    def test_local_launcher_preflight_does_not_contact_providers(self):
        from northstar_atlas import live_preflight
        from contextlib import redirect_stdout
        import io
        _, owner = self.enable_runtime()
        with patch('northstar_atlas.live_budget.configured_budget', return_value=self.budget), \
             patch('pathlib.Path.cwd', return_value=self.root), \
             patch.dict(os.environ, VECTOR_STORE='pinecone'), \
             patch.object(live_preflight, 'require_secret_environment') as secrets, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(live_preflight.main(), 0)
            secrets.assert_called_once_with('pinecone')
        self.assertEqual(len(self.sent), 0)
        self.assertEqual(len(self.created), 0)
        self.assertEqual(owner.batch_remaining(), 5)
