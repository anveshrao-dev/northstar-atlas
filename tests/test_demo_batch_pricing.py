"""Unused-batch renewal: synthetic authorities and fake providers only."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import importlib
import json
import multiprocessing
import os
import unittest
from unittest.mock import patch

import test_demo_batch as fixture_module
from budget_fixture import Provider, request
from northstar_atlas import demo_batch as batch
from northstar_atlas.budget_transition import _encode
from northstar_atlas.diagnostic_window import sha, WINDOW_NAME
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget


class RenewalTests(unittest.TestCase):
    def setUp(self):
        f = fixture_module.BatchTests('runTest'); f.setUp(); self.addCleanup(f.doCleanups)
        first, budget = f.activate()
        for i in range(5):
            if i == 2:
                with self.assertRaises(TimeoutError):
                    f.invoke(budget, first.plan_sha256, i, TimeoutError())
            else:
                f.invoke(budget, first.plan_sha256, i)
        f.budget = budget; f.previous = first.plan_sha256
        f.contract = batch.batch_contract(budget, 'second-batch')
        original = f.prepare(approved_max_attempts=12); f.apply(original)
        self.f = f; self.previous = original.plan_sha256
        self.original_directory = original.artifact_directory
        self.root, self.path = f.root, f.path
        self.now = budget.clock()+timedelta(days=2)
        self.budget = PersistentBudget(self.root, scope=budget.scope, billing_scope=budget.billing_scope,
                                      policy=BudgetPolicy(12), clock=lambda: self.now)
        self.source = self.path.read_bytes(); self.history = json.loads(self.source)
        self.window = (self.path.parent/WINDOW_NAME).read_bytes()
        self.price = deepcopy(self.history['pricing_revisions'][0])
        self.price.update(id='renewed-price', confirmed_at=self.now.isoformat(),
                          valid_until=(self.now+timedelta(hours=1)).isoformat())

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.demo_batch_pricing'),
                             'unused-batch pricing renewal is not implemented')
        return importlib.import_module('northstar_atlas.demo_batch_pricing')

    def prepare(self, **changes):
        options = dict(previous_plan_sha256=self.previous,
            confirmed_billing_scope=self.budget.billing_scope, writers_stopped=True,
            verified_pricing_sha256=sha(_encode(self.price)), dry_run=False)
        options.update(changes)
        return self.module().prepare_renewal(self.budget, self.price, **options)

    def apply(self, preview, **changes):
        options = dict(previous_plan_sha256=self.previous, writers_stopped=True,
                       approved_plan_sha256=preview.plan_sha256)
        options.update(changes)
        return self.module().apply_renewal(self.budget, preview.plan_sha256, **options)

    def invoke(self, plan, index, failure=None):
        spec = replace(self.f.spec, invocation_id=f'renewal-{index}',
                       mode='draft_review' if index % 2 else 'approved_only')
        with self.budget.reserve(spec, diagnostic_approval=plan) as permit:
            permit.invoke(Provider(failure), 'generator', 0, request())

    def test_dry_run_history_and_private_artifacts(self):
        before = set(self.path.parent.iterdir())
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(set(self.path.parent.iterdir()), before)
        self.assertEqual(self.path.read_bytes(), self.source)
        original_files = {p.name:p.read_bytes() for p in self.original_directory.iterdir()}
        saved = self.prepare(); self.assertEqual(saved.plan_sha256, preview.plan_sha256)
        directory = saved.artifact_directory
        self.assertEqual((directory/'backup.v2.json').read_bytes(), self.source)
        self.assertEqual((directory/'backup.window.json').read_bytes(), self.window)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in directory.iterdir()))
        self.apply(saved)
        expected = deepcopy(self.history)
        expected['pricing_revisions'].append(self.price); expected['active_pricing'] = self.price['id']
        self.assertEqual(self.path.read_bytes(), _encode(expected))
        self.assertEqual(sha(self.path.read_bytes()), saved.candidate_sha256)
        self.assertEqual(original_files, {p.name:p.read_bytes() for p in self.original_directory.iterdir()})
        self.assertEqual(self.budget.inspect_history(diagnostic_approval=saved.plan_sha256).attempts_consumed, 7)
        self.assertEqual(expected['workflows'][4]['status'], 'uncertain')

    def test_five_submissions_duplicates_restart_and_sixth_denial(self):
        saved = self.prepare(); self.apply(saved)
        for i in range(5):
            self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 5-i)
            self.invoke(saved.plan_sha256, i)
            with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, i)
            old = self.budget
            self.budget = PersistentBudget(self.root, scope=old.scope, billing_scope=old.billing_scope,
                                          policy=BudgetPolicy(12), clock=lambda: self.now)
        self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 0)
        with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, 5)
        current = json.loads(self.path.read_bytes())
        self.assertEqual(current['workflows'][:7], self.history['workflows'])
        self.assertEqual(len(current['workflows']), 12)

    def test_old_approval_and_repeated_renewal_rejected(self):
        saved = self.prepare(); self.apply(saved)
        with self.assertRaises(BudgetError): batch.remaining(self.budget, self.previous)
        with self.assertRaises(BudgetError): self.invoke(self.previous, 0)
        with self.assertRaises(BudgetError): self.prepare(previous_plan_sha256=saved.plan_sha256)
        with self.assertRaises(BudgetError): self.apply(saved)

    def test_expired_future_duplicate_and_unverified_pricing(self):
        original = deepcopy(self.price)
        for fields in ({'valid_until':self.now.isoformat()},
                       {'confirmed_at':(self.now+timedelta(minutes=1)).isoformat()},
                       {'valid_until':(self.now+timedelta(hours=25)).isoformat()},
                       {'id':self.history['active_pricing']},
                       {'source':'https://unreviewed.invalid/prices'},
                       {'input_per_million_usd':'100'}):
            self.price = dict(original, **fields)
            with self.subTest(fields=fields), self.assertRaises(BudgetError): self.prepare()
        self.price = original
        with self.assertRaises(BudgetError): self.prepare(verified_pricing_sha256='0'*64)

    def test_expiry_between_prepare_apply_and_during_runtime(self):
        saved = self.prepare(); self.now += timedelta(hours=2)
        with self.assertRaises(BudgetError): self.apply(saved)
        self.assertEqual(self.path.read_bytes(), self.source)
        self.now -= timedelta(hours=2); self.apply(saved); self.now += timedelta(hours=2)
        with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, 0)

    def test_authorization_and_identity_tampering(self):
        for change in (dict(writers_stopped=False), dict(confirmed_billing_scope='wrong')):
            with self.assertRaises(BudgetError): self.prepare(**change)
        saved = self.prepare()
        with self.assertRaises(BudgetError): self.apply(saved, approved_plan_sha256='0'*64)
        from northstar_atlas import diagnostic_window
        original = diagnostic_window.identity
        with patch.object(diagnostic_window, 'identity', side_effect=lambda p:
                          [original(p)[0]+1, original(p)[1]]):
            with self.assertRaises(BudgetError): self.apply(saved)

    def test_changed_source_and_candidate_bytes(self):
        saved = self.prepare()
        self.path.write_bytes(self.source+b' ')
        with self.assertRaises(BudgetError): self.apply(saved)
        self.path.write_bytes(self.source)
        path = saved.artifact_directory/'candidate.v2.json'; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(saved)

    def test_partially_consumed_original_batch_rejected(self):
        self.now -= timedelta(days=2)
        self.invoke(self.previous, 0)
        self.now += timedelta(days=2)
        with self.assertRaises(BudgetError): self.prepare()

    def test_pending_settlement_denied(self):
        self.budget._mark_settlement()
        with self.assertRaises(BudgetError): self.prepare()

    def test_partial_apply_is_guarded_without_automatic_rollback(self):
        saved = self.prepare()
        with patch.object(self.budget, '_write', side_effect=OSError('synthetic')):
            with self.assertRaises(BudgetError): self.apply(saved)
        self.assertEqual(self.path.read_bytes(), self.source)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertNotEqual((self.path.parent/WINDOW_NAME).read_bytes(), self.window)
        with self.assertRaises(BudgetError): self.apply(saved)
        with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, 0)

    def test_uncertain_submission_preserves_full_reservation(self):
        saved = self.prepare(); self.apply(saved)
        with self.assertRaises(TimeoutError): self.invoke(saved.plan_sha256, 0, TimeoutError())
        self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 4)
        current = json.loads(self.path.read_bytes())
        self.assertEqual(current['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(current['workflows'][:7], self.history['workflows'])

    def test_lock_contention_in_other_process(self):
        ctx = multiprocessing.get_context('fork')
        parent, child = ctx.Pipe()
        def attempt():
            try: self.prepare(dry_run=True)
            except BudgetError: child.send('denied')
            else: child.send('unsafe')
        with self.budget._locked(diagnostic_approval=self.previous):
            process = ctx.Process(target=attempt); process.start()
            self.assertTrue(parent.poll(10)); self.assertEqual(parent.recv(), 'denied')
            process.join(10); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()

    def test_changed_lock_and_ledger_inodes_rejected(self):
        saved = self.prepare()
        replacement = self.path.parent/'replacement.json'
        replacement.write_bytes(self.source); replacement.chmod(0o600)
        os.replace(replacement, self.path)
        with self.assertRaises(BudgetError): self.apply(saved)
        self.budget.lock_path.rename(self.path.parent/'retired.lock')
        self.budget.lock_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.prepare(dry_run=True)

    def test_missing_authority_is_never_initialized(self):
        self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.path.exists())
        self.assertEqual((self.path.parent/WINDOW_NAME).read_bytes(), self.window)

    def test_private_permissions_and_symlink_rejected(self):
        saved = self.prepare()
        candidate = saved.artifact_directory/'candidate.v2.json'
        candidate.chmod(0o644)
        with self.assertRaises(BudgetError): self.apply(saved)
        candidate.chmod(0o600)
        backup = saved.artifact_directory/'backup.v2.json'
        backup.unlink(); backup.symlink_to(self.path)
        with self.assertRaises(BudgetError): self.apply(saved)

    def test_readback_failure_preserves_guard_and_new_bytes(self):
        saved = self.prepare()
        with patch.object(self.budget, '_read', side_effect=BudgetError('synthetic readback')):
            with self.assertRaises(BudgetError): self.apply(saved)
        self.assertEqual(sha(self.path.read_bytes()), saved.candidate_sha256)
        self.assertTrue(self.budget.pending_path.exists())
        fresh = PersistentBudget(self.root, scope=self.budget.scope, billing_scope=self.budget.billing_scope,
                                 policy=BudgetPolicy(12), clock=lambda: self.now)
        with self.assertRaises(BudgetError): batch.remaining(fresh, saved.plan_sha256)

    def test_zero_call_failures_still_consume_five_submissions(self):
        saved = self.prepare(); self.apply(saved)
        for i in range(5):
            with self.budget.reserve(replace(self.f.spec, invocation_id=f'empty-{i}'),
                                     diagnostic_approval=saved.plan_sha256): pass
        self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 0)
        with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, 9)

    def test_wrong_backend_profile_mode_and_corpus_rejected(self):
        from northstar_atlas.live_budget import GRAPH
        saved = self.prepare(); self.apply(saved)
        for changes in (dict(profile=GRAPH), dict(backend='chroma'), dict(mode='all')):
            with self.subTest(changes=changes), self.assertRaises(BudgetError):
                with self.budget.reserve(replace(self.f.spec, **changes),
                                         diagnostic_approval=saved.plan_sha256): self.fail('unauthorized')
        with self.assertRaises(BudgetError):
            with self.budget.reserve(self.f.spec): self.fail('missing approval')
        manifest = self.root/'data/corpus_manifest.json'
        manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.invoke(saved.plan_sha256, 0)

    def runtime(self):
        import test_linear_demo
        from test_context import chunk
        from northstar_atlas.ingestion.document_loader import APPROVED
        from northstar_atlas.ui.runtime import ResourceOwner
        saved = self.prepare(); self.apply(saved)
        access = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.budget.before_request = access
        self.f.budget = self.budget
        self.enterContext(patch.dict(os.environ, NORTHSTAR_CONTROLLED_LIVE='1',
            NORTHSTAR_DEMO_BATCH_ENABLED='1', NORTHSTAR_DEMO_BATCH_PLAN=saved.plan_sha256,
            NORTHSTAR_DEMO_BATCH_RUN_APPROVAL=batch.run_approval(saved.plan_sha256),
            NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1', NORTHSTAR_UI_ENABLE_GRAPH='0'))
        self.enterContext(patch('test_linear_demo.chunk', return_value=chunk(status=APPROVED,
            effective=self.f.spec.reference_date)))
        test_linear_demo.LinearDemoTests.wire(self.f)
        owner = ResourceOwner(self.root, live_enabled=True, pinecone_enabled=True, budget=self.budget)
        self.addCleanup(owner.close)
        return saved, owner, access

    def test_renewed_runtime_settles_and_captures_before_display(self):
        from northstar_atlas.ui import runtime
        saved, owner, access = self.runtime()
        original_project = runtime.project_result
        def project(result, mode):
            record = self.root/'.local/linear-demo-results/renewed-ui.json'
            self.assertTrue(record.exists())
            self.assertEqual(json.loads(self.path.read_bytes())['workflows'][-1]['status'], 'closed')
            return original_project(result, mode)
        submission = self.f.submission('renewed-ui')
        with patch.object(runtime, 'project_result', side_effect=project):
            result = owner.submit(submission, consent=True)
        self.assertEqual(result.kind, 'answer', result)
        self.assertTrue(result.citations); self.assertGreaterEqual(access.call_count, 2)
        self.assertEqual(len(self.f.sent), 1)
        self.assertEqual(owner.submit(submission, consent=True).code, 'duplicate')
        self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 4)

    def test_renewed_runtime_access_and_settlement_fail_closed(self):
        saved, owner, access = self.runtime()
        access.side_effect = ValueError('private fixture')
        self.assertEqual(owner.submit(self.f.submission('denied'), consent=True).kind, 'error')
        self.assertEqual(len(self.f.sent), 0)
        access.side_effect = None
        with patch.object(self.budget, '_settle', side_effect=BudgetError('synthetic')):
            result = owner.submit(self.f.submission('unsettled'), consent=True)
        self.assertEqual(result.kind, 'error'); self.assertEqual(len(self.f.sent), 1)
        record = (self.root/'.local/linear-demo-results/unsettled.json').read_bytes()
        self.assertNotIn(b'accepted', record)
        self.assertNotIn(b'private fixture', record)

    def test_cli_dry_run_and_separate_apply_approval(self):
        import io
        from contextlib import redirect_stdout
        module = self.module()
        price_file = self.path.parent/'reviewed-price.json'
        price_file.write_bytes(_encode(self.price)); price_file.chmod(0o600)
        common = ['--root', str(self.root)]
        prepare = common+['prepare', '--pricing', str(price_file), '--verified-pricing-sha256',
            sha(_encode(self.price)), '--confirmed-billing-scope', self.budget.billing_scope,
            '--previous-plan', self.previous, '--writers-stopped']
        with patch.object(module, 'configured_budget', return_value=self.budget), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(module.main(prepare), 0)
            preview = json.loads(out.getvalue())
        self.assertEqual(self.path.read_bytes(), self.source)
        with patch.object(module, 'configured_budget', return_value=self.budget), redirect_stdout(io.StringIO()):
            self.assertEqual(module.main(prepare+['--write-artifacts']), 0)
            args = common+['apply', '--plan', preview['plan_sha256'], '--previous-plan', self.previous,
                           '--writers-stopped', '--approve']
            self.assertEqual(module.main(args+['0'*64]), 1)
            self.assertEqual(module.main(args+[preview['plan_sha256']]), 0)

    def test_concurrent_duplicate_submission_is_accounted_once(self):
        saved = self.prepare(); self.apply(saved)
        ctx = multiprocessing.get_context('fork')
        parent, child = ctx.Pipe()
        def attempt():
            try: self.invoke(saved.plan_sha256, 0)
            except BudgetError: child.send('denied')
            else: child.send('reserved')
        processes = [ctx.Process(target=attempt) for _ in range(2)]
        for process in processes: process.start()
        results = []
        for _ in processes:
            self.assertTrue(parent.poll(10)); results.append(parent.recv())
        for process in processes:
            process.join(10); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(sorted(results), ['denied', 'reserved'])
        self.assertEqual(batch.remaining(self.budget, saved.plan_sha256), 4)

    def test_original_approval_and_candidate_tampering_rejected(self):
        saved = self.prepare()
        backup = saved.artifact_directory/'backup.window.json'
        backup.write_bytes(backup.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(saved)
        backup.write_bytes(self.window)
        original = self.original_directory/'candidate.v2.json'
        original.write_bytes(original.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(saved)

    def test_pending_workflow_and_nonseven_history_rejected(self):
        module = self.module()
        previous = json.loads(self.window)
        for change in ('pending', 'count'):
            value = deepcopy(self.history)
            if change == 'pending':
                value['workflows'][-1].update(status='reserved', finished_at=None)
            else:
                value['workflows'].pop()
            raw = _encode(value)
            modified = dict(previous, candidate_sha256=sha(raw))
            with patch.object(module, 'validate_plan'), patch.object(module, '_private_directory'), \
                 patch.object(module, '_read_private', return_value=raw):
                with self.assertRaises(BudgetError): module._candidate(self.budget, raw, modified, self.price)
