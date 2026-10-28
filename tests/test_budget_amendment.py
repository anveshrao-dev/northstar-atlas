"""Synthetic v2 amendments only. All sockets are blocked."""
from copy import deepcopy
from contextlib import contextmanager, redirect_stdout
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import io
import json
import multiprocessing
import os
import socket
import unittest
from unittest.mock import patch

from budget_fixture import BILLING, SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from test_budget_transition import history
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH, LINEAR
from northstar_atlas import budget_amendment as amendment


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def child_apply(root, plan, send, crash=False):
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(8))
    if crash:
        original = budget._write
        def write(value):
            original(value)
            os._exit(0)
        budget._write = write
    try:
        amendment.apply_amendment(budget, plan, approved_plan_sha256=plan, writers_stopped=True)
    except BudgetError:
        send.send('blocked')
    else:
        send.send('applied')


class AmendmentTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root),
                                     NORTHSTAR_CONTROLLED_LIVE='0'))
        for key in ('NORTHSTAR_BUDGET_AUTHORITY_ROOT', 'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED'):
            os.environ.pop(key, None)
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.path = install(self.root, document(calls=8, legacy=history()))
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(8))
        old = WorkflowRequest('prior-uncertain', GRAPH, 'Prior question', 'draft_review', 'chroma', date(2026, 10, 27))
        with self.assertRaises(TimeoutError):
            with self.budget.reserve(old) as permit:
                permit.invoke(Provider(TimeoutError('private detail')), 'router', 0, request(256, 8192))
        self.original_value = json.loads(self.path.read_bytes())
        self.original = (json.dumps(self.original_value, indent=3) + '\n').encode()
        self.path.write_bytes(self.original)
        self.spec = WorkflowRequest('approved-retest', GRAPH, 'Who?', 'draft_review', 'chroma', date(2026, 10, 27))
        self.contract = amendment.request_contract(self.budget, self.spec)

    def prepare(self, **changes):
        options = dict(confirmed_billing_scope=BILLING, approved_max_attempts=15,
                       approved_max_reserved_usd='0.100000', writers_stopped=True, dry_run=False)
        options.update(changes)
        return amendment.prepare_amendment(self.budget, self.contract, **options)

    def apply(self, preview, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256, writers_stopped=True)
        options.update(changes)
        return amendment.apply_amendment(self.budget, preview.plan_sha256, **options)

    def updated(self):
        return PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(15))

    def test_dry_run_no_writes_and_private_exact_backup(self):
        before = {p.name: p.read_bytes() for p in self.path.parent.iterdir()}
        preview = self.prepare(dry_run=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.parent.iterdir()})
        self.assertEqual(preview.attempts_preserved, 8)
        self.assertIsNone(preview.artifact_directory)
        saved = self.prepare()
        self.assertEqual(saved.plan_sha256, preview.plan_sha256)
        self.assertEqual((saved.artifact_directory/'backup.v2.json').read_bytes(), self.original)
        self.assertEqual(saved.artifact_directory.stat().st_mode & 0o777, 0o700)
        for p in saved.artifact_directory.iterdir():
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_exact_amendment_history_and_cost_conserved(self):
        before = self.budget.inspect_history()
        preview = self.prepare(); self.apply(preview)
        expected = deepcopy(self.original_value); expected['max_attempts'] = 15
        expected['diagnostic_request_sha256'] = sha(encode(self.contract))
        self.assertEqual(json.loads(self.path.read_bytes()), expected)
        after = self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)
        self.assertEqual(after, before)
        self.assertEqual(self.path.with_name('budget.lock').stat().st_ino, self.budget.lock_path.stat().st_ino)
        with self.assertRaises(BudgetError): self.updated().check_ready()
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_append_new_price_without_repricing_history(self):
        price = document()['pricing_revisions'][0]; price['id'] = 'reviewed-new-price'
        price['output_per_million_usd'] = '0.4'
        preview = self.prepare(pricing_revision=price, verified_pricing_sha256=sha(encode(price)))
        self.apply(preview)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['pricing_revisions'][:-1], self.original_value['pricing_revisions'])
        self.assertEqual(value['workflows'], self.original_value['workflows'])
        self.assertEqual(value['legacy'], self.original_value['legacy'])
        self.assertEqual(value['active_pricing'], price['id'])

    def test_wrong_caps_scope_approval_or_price_rejected(self):
        bad_options = [dict(approved_max_attempts=n) for n in (8, 14, 16)] + [
            dict(approved_max_reserved_usd='0.2'), dict(confirmed_billing_scope='other'),
            dict(writers_stopped=False), dict(verified_pricing_sha256='0'*64),
            dict(pricing_revision=document()['pricing_revisions'][0], verified_pricing_sha256='0'*64)]
        for options in bad_options:
            with self.subTest(options=options), self.assertRaises(BudgetError): self.prepare(**options)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_expired_price_and_expiry_between_prepare_apply(self):
        preview = self.prepare()
        now = self.budget.clock(); self.budget.clock = lambda: now + timedelta(hours=2)
        with self.assertRaises(BudgetError): self.apply(preview)
        with self.assertRaises(BudgetError): self.prepare(dry_run=True)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_candidate_source_and_approval_binding(self):
        preview = self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview, approved_plan_sha256='0'*64)
        candidate = preview.artifact_directory/'candidate.v2.json'
        original = candidate.read_bytes(); candidate.write_bytes(original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        candidate.write_bytes(original)
        self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.path.write_bytes(self.original)
        self.apply(preview)

    def test_lock_identity_change_rejected(self):
        preview = self.prepare()
        self.budget.lock_path.rename(self.budget.directory/'old-lock')
        self.budget.lock_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_lock_contention_across_processes(self):
        preview = self.prepare(); ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        with self.budget._locked():
            process = ctx.Process(target=child_apply, args=(self.root, preview.plan_sha256, child))
            process.start(); self.assertTrue(parent.poll(15)); self.assertEqual(parent.recv(), 'blocked')
            process.join(15); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_failed_write_and_crash_leave_guard_no_automatic_restore(self):
        preview = self.prepare()
        with patch.object(self.budget, '_write', side_effect=OSError('private failure')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_crash_after_replacement_blocks_restart(self):
        preview = self.prepare(); ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        process = ctx.Process(target=child_apply, args=(self.root, preview.plan_sha256, child, True))
        process.start(); process.join(15); self.assertEqual(process.exitcode, 0)
        parent.close(); child.close()
        self.assertEqual(json.loads(self.path.read_bytes())['max_attempts'], 15)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)
        self.assertEqual((preview.artifact_directory/'backup.v2.json').read_bytes(), self.original)

    def test_window_permits_only_one_bound_invocation_even_with_zero_calls(self):
        preview = self.prepare(); self.apply(preview); budget = self.updated()
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec): pass
        changed = WorkflowRequest('other', GRAPH, 'Who?', 'draft_review', 'chroma', self.spec.reference_date)
        with self.assertRaises(BudgetError):
            with budget.reserve(changed, diagnostic_approval=preview.plan_sha256): pass
        with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        linear = WorkflowRequest('linear', LINEAR, 'Who?', 'draft_review', 'chroma', self.spec.reference_date)
        with self.assertRaises(BudgetError):
            with self.updated().reserve(linear): pass
        self.assertEqual(len(json.loads(self.path.read_bytes())['workflows']), 2)

    def test_corruption_missing_ledger_and_pending_state_never_initialize(self):
        for bad in ({}, history(), dict(self.original_value, workflows=[])):
            self.path.write_bytes(encode(bad))
            with self.assertRaises(BudgetError): self.prepare()
        self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.path.exists())

    def test_pending_workflow_and_settlement_guard_fail_closed(self):
        value = deepcopy(self.original_value)
        value['workflows'][0].update(status='reserved', finished_at=None)
        self.path.write_bytes(encode(value))
        with self.assertRaises(BudgetError): self.prepare()
        self.path.write_bytes(self.original)
        self.budget.pending_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_invalid_price_revision_duration_duplicate_and_cost_exhaustion(self):
        from datetime import datetime
        for variant in ('too-long', 'same-id', 'too-expensive', 'wrong-digest'):
            price = document()['pricing_revisions'][0]
            if variant != 'same-id': price['id'] = 'new-price'
            if variant == 'too-long':
                price['valid_until'] = (datetime.fromisoformat(price['confirmed_at'])+timedelta(hours=25)).isoformat()
            if variant == 'too-expensive': price['input_per_million_usd'] = '1000'
            proof = '0'*64 if variant == 'wrong-digest' else sha(encode(price))
            with self.subTest(variant=variant), self.assertRaises(BudgetError):
                self.prepare(pricing_revision=price, verified_pricing_sha256=proof)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_expired_history_can_append_current_price_without_refund(self):
        now = self.budget.clock(); self.budget.clock = lambda: now+timedelta(hours=2)
        price = document()['pricing_revisions'][0]
        price.update(id='fresh', confirmed_at=(now+timedelta(hours=2)).isoformat(),
                     valid_until=(now+timedelta(hours=3)).isoformat())
        preview = self.prepare(pricing_revision=price, verified_pricing_sha256=sha(encode(price)))
        self.apply(preview)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'], self.original_value['workflows'])
        self.assertEqual(value['pricing_revisions'][0], self.original_value['pricing_revisions'][0])

    def test_candidate_cannot_remove_uncertainty_or_lower_reserved_history(self):
        preview = self.prepare()
        path = preview.artifact_directory/'candidate.v2.json'
        value = json.loads(path.read_bytes())
        value['workflows'] = []
        path.write_bytes(encode(value))
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_replaced_source_inode_invalidates_approval_even_identical_bytes(self):
        preview = self.prepare()
        replacement = self.path.with_name('synthetic-replacement')
        replacement.write_bytes(self.original); replacement.chmod(0o600)
        replacement.replace(self.path)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_artifact_and_window_symlinks_or_permissions_rejected(self):
        preview = self.prepare()
        backup = preview.artifact_directory/'backup.v2.json'
        backup.chmod(0o644)
        with self.assertRaises(BudgetError): self.apply(preview)
        backup.chmod(0o600); self.apply(preview)
        window = self.path.parent/'diagnostic-window.json'
        window.chmod(0o644)
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        window.chmod(0o600)
        saved = self.path.parent/'saved-window'
        window.rename(saved); window.symlink_to(saved)
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass

    def test_window_tamper_or_missing_window_cannot_authorize_request(self):
        preview = self.prepare(); self.apply(preview)
        window = self.path.parent/'diagnostic-window.json'
        window.write_bytes(window.read_bytes()+b' ')
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        window.unlink()
        with self.assertRaises(BudgetError):
            with self.updated().reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        linear = WorkflowRequest('ordinary', LINEAR, 'Who?', 'draft_review', 'chroma', self.spec.reference_date)
        with self.assertRaises(BudgetError):
            with self.updated().reserve(linear): pass
        with self.assertRaises(BudgetError): self.updated().check_ready()

    def test_readback_failure_never_removes_settlement_guard(self):
        preview = self.prepare(); original_write = self.budget._write
        def write(value):
            original_write(value)
            self.path.write_bytes(self.path.read_bytes()+b' ')
        with patch.object(self.budget, '_write', side_effect=write):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_apply_fsync_failure_is_uncertain_even_after_replacement(self):
        preview = self.prepare(); original_write = self.budget._write
        def write(value):
            original_write(value)
            raise OSError('simulated post-replace fsync failure')
        with patch.object(self.budget, '_write', side_effect=write):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(json.loads(self.path.read_bytes())['max_attempts'], 15)

    def test_manifest_change_before_lock_rejected_without_reservation(self):
        preview = self.prepare(); self.apply(preview); budget = self.updated()
        locked = budget._locked
        manifest = self.root/'data/corpus_manifest.json'
        @contextmanager
        def changed(**kwargs):
            with locked(**kwargs) as plan:
                # Still valid JSON with the same source hashes, but new bytes.
                manifest.write_bytes(manifest.read_bytes()+b'\n')
                yield plan
        before = self.path.read_bytes()
        with patch.object(budget, '_locked', side_effect=changed), self.assertRaises(BudgetError):
            with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_preparation_defaults_to_dry_run_without_private_identity_output(self):
        contract = self.root/'.local/request-contract.json'
        contract.write_bytes(encode(self.contract)); contract.chmod(0o600)
        before = {p.name:p.read_bytes() for p in self.path.parent.iterdir()}
        output = io.StringIO()
        with patch.dict(os.environ, NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
                        NORTHSTAR_BUDGET_MAX_ATTEMPTS='8'), redirect_stdout(output):
            status = amendment.main(['--root', str(self.root), 'prepare', '--contract', str(contract),
                '--confirmed-billing-scope', BILLING, '--approve-max-attempts', '15',
                '--approve-max-reserved-usd', '0.100000', '--writers-stopped'])
        self.assertEqual(status, 0)
        self.assertEqual(before, {p.name:p.read_bytes() for p in self.path.parent.iterdir()})
        for private in (BILLING, str(self.root), self.spec.question): self.assertNotIn(private, output.getvalue())
