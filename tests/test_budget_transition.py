"""Offline transitions of synthetic ledgers only; never real operator history."""
import importlib.util
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest
from unittest.mock import patch

from budget_fixture import BILLING, SCOPE, document, install
from checkout_fixture import ROOT, checkout
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, LINEAR


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def history(status='succeeded'):
    old = datetime.now(timezone.utc) - timedelta(days=2)
    pricing = dict(input_per_million_usd='0.1', output_per_million_usd='0.3',
        source='https://nebius.com/prices', confirmed_at=old.isoformat(),
        valid_until=(old+timedelta(hours=1)).isoformat())
    return dict(version=1, scope=SCOPE, model=document()['model'], max_attempts=3,
        max_reserved_usd='0.100000', pricing=pricing, attempts=[dict(id='2'*32,
        reserved_at=old.isoformat(), finished_at=None if status=='reserved' else (old+timedelta(minutes=1)).isoformat(),
        input_tokens=1000, output_tokens=100, reserved_usd='0.000130', status=status)])


def child_apply(root, plan_id, send=None, crash=False):
    from northstar_atlas.budget_transition import apply_transition
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(max_attempts=8))
    if crash:
        original = budget._write
        def terminate_after_replace(value):
            original(value)
            os._exit(0)
        budget._write = terminate_after_replace
    try:
        apply_transition(budget, plan_id, approved_plan_sha256=plan_id, writers_stopped=True)
    except BudgetError:
        if send is not None: send.send('blocked')
    else:
        if send is not None: send.send('applied')


class TransitionAvailability(unittest.TestCase):
    def test_operator_transition_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.budget_transition'))


class TransitionTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec('northstar_atlas.budget_transition') is None:
            self.skipTest('transition implementation absent')
        from northstar_atlas import budget_transition
        self.t = budget_transition
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root),
            NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
            NORTHSTAR_BUDGET_MAX_ATTEMPTS='8', NORTHSTAR_CONTROLLED_LIVE='0'))
        for name in ('NORTHSTAR_BUDGET_AUTHORITY_ROOT','NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED'):
            os.environ.pop(name, None)
        self.enterContext(patch.object(socket.socket,'connect',side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket,'connect_ex',side_effect=AssertionError('network blocked')))
        self.legacy = history()
        self.path = install(self.root, self.legacy)
        # Deliberately noncanonical JSON so byte conservation is meaningful.
        self.original = (json.dumps(self.legacy, indent=3) + '\n').encode()
        self.path.write_bytes(self.original)
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                      policy=BudgetPolicy(max_attempts=8))
        self.price = document()['pricing_revisions'][0]

    def prepare(self, *, dry_run=False, **changes):
        options=dict(confirmed_billing_scope=BILLING, approved_max_attempts=8,
            approved_max_reserved_usd='0.100000', verified_pricing_sha256=digest(encoded(self.price)),
            writers_stopped=True, dry_run=dry_run)
        options.update(changes)
        return self.t.prepare_transition(self.budget, self.price, **options)

    def apply(self, preview):
        return self.t.apply_transition(self.budget, preview.plan_sha256,
            approved_plan_sha256=preview.plan_sha256, writers_stopped=True)

    def rollback(self, preview):
        return self.t.rollback_transition(self.budget, preview.plan_sha256,
            approved_rollback_sha256=digest(('rollback:'+preview.plan_sha256).encode()), writers_stopped=True)

    def assert_original(self):
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_dry_run_validates_candidate_without_any_filesystem_write(self):
        before={str(p.relative_to(self.path.parent)):p.read_bytes() for p in self.path.parent.iterdir()}
        preview=self.prepare(dry_run=True)
        self.assertEqual(preview.attempts_preserved,1)
        self.assertEqual(preview.reserved_usd,'0.000130')
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(before,{str(p.relative_to(self.path.parent)):p.read_bytes() for p in self.path.parent.iterdir()})
        prepared=self.prepare()
        self.assertEqual(prepared.plan_sha256,preview.plan_sha256)

    def test_prepare_private_exact_backup_and_full_unrepriced_history(self):
        preview=self.prepare(); folder=preview.artifact_directory
        self.assertEqual((folder/'backup.v1.json').read_bytes(),self.original)
        self.assertEqual(digest(self.original),preview.source_sha256)
        candidate=json.loads((folder/'candidate.v2.json').read_bytes())
        self.assertEqual(candidate['legacy'],self.legacy)
        self.assertEqual(candidate['workflows'],[])
        self.assertEqual(candidate['billing_scope'],BILLING)
        self.assertEqual(candidate['pricing_revisions'],[self.price])
        self.assertEqual(folder.stat().st_mode & 0o777,0o700)
        for path in folder.iterdir():self.assertEqual(path.stat().st_mode & 0o777,0o600)
        self.assert_original()

    def test_apply_atomic_readback_preserves_history_counts_and_lock_inode(self):
        preview=self.prepare(); lock_before=self.budget.lock_path.stat().st_ino
        self.apply(preview)
        state=self.budget.check_ready()
        self.assertEqual((state.version,state.attempts_consumed,state.reserved_usd),(2,1,Decimal('0.000130')))
        self.assertEqual(self.budget.lock_path.stat().st_ino,lock_before)
        self.assertEqual(digest(self.path.read_bytes()),preview.candidate_sha256)
        self.assertFalse(self.budget.pending_path.exists())
        self.assertTrue((preview.artifact_directory/'apply.completed').is_file())

    def test_finished_uncertain_history_is_preserved_with_full_reserved_charge(self):
        self.legacy=history('failed_or_uncertain'); self.path.write_bytes(encoded(self.legacy))
        preview=self.prepare();self.apply(preview)
        self.assertEqual(json.loads(self.path.read_bytes())['legacy'],self.legacy)
        self.assertEqual(self.budget.check_ready().reserved_usd,Decimal('0.000130'))

    def test_pending_attempt_blocks_prepare_without_backup_or_reinterpretation(self):
        self.path.write_bytes(encoded(history('reserved')))
        before=self.path.read_bytes()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual({p.name for p in self.path.parent.iterdir()},{'budget.json','budget.lock'})

    def test_writer_attestation_required_for_each_operation(self):
        with self.assertRaises(BudgetError):self.prepare(writers_stopped=False)
        preview=self.prepare()
        with self.assertRaises(BudgetError):self.t.apply_transition(self.budget,preview.plan_sha256,
            approved_plan_sha256=preview.plan_sha256,writers_stopped=False)
        self.assert_original()

    def test_billing_limits_and_pricing_need_independent_matching_confirmation(self):
        cases=[dict(confirmed_billing_scope='another-scope'),dict(approved_max_attempts=7),
               dict(approved_max_attempts=True),dict(approved_max_reserved_usd='0.09'),
               dict(verified_pricing_sha256='0'*64)]
        for change in cases:
            with self.subTest(change=change),self.assertRaises(BudgetError):self.prepare(**change)
        self.assert_original()

    def test_expired_future_overlong_or_invalid_pricing_is_rejected(self):
        original=self.price.copy();now=datetime.now(timezone.utc)
        cases=[dict(valid_until=(now-timedelta(minutes=1)).isoformat()),
               dict(confirmed_at=(now+timedelta(minutes=1)).isoformat()),
               dict(valid_until=(now+timedelta(hours=25)).isoformat()),
               dict(source='https://unverified.invalid/prices'),dict(input_per_million_usd='0')]
        for change in cases:
            self.price=original|change
            with self.subTest(change=change),self.assertRaises(BudgetError):self.prepare()
        self.assert_original()

    def test_corrupt_unknown_duplicate_or_conflicting_history_fails_closed(self):
        wrong_scope=self.legacy|{'scope':'3'*32}
        duplicate=self.legacy|{'attempts':self.legacy['attempts']*2}
        cost=self.legacy|{'attempts':[self.legacy['attempts'][0]|{'reserved_usd':'0.000001'}]}
        for raw in (b'{}',b'{"version":1,"version":1}',encoded(wrong_scope),encoded(duplicate),encoded(cost),encoded(document(calls=8))):
            self.path.write_bytes(raw)
            with self.subTest(raw=raw[:16]),self.assertRaises(BudgetError):self.prepare()
            self.assertEqual(self.path.read_bytes(),raw)

    def test_missing_ledger_and_invalid_permissions_never_create_another_ledger(self):
        self.path.chmod(0o644)
        with self.assertRaises(BudgetError):self.prepare()
        self.path.chmod(0o600);self.path.unlink()
        with self.assertRaises(BudgetError):self.prepare()
        self.assertEqual({p.name for p in self.path.parent.iterdir()},{'budget.lock'})

    def test_candidate_digest_approval_is_required_before_mutation(self):
        preview=self.prepare()
        with self.assertRaises(BudgetError):self.t.apply_transition(self.budget,preview.plan_sha256,
            approved_plan_sha256='0'*64,writers_stopped=True)
        self.assert_original()
        self.assertFalse((preview.artifact_directory/'apply.started').exists())

    def test_tampered_backup_candidate_or_manifest_rejected_before_mutation(self):
        preview=self.prepare()
        for name in ('backup.v1.json','candidate.v2.json','plan.json'):
            path=preview.artifact_directory/name;raw=path.read_bytes();path.write_bytes(raw+b' ')
            try:
                with self.subTest(name=name),self.assertRaises(BudgetError):self.apply(preview)
            finally:path.write_bytes(raw)
        self.assert_original()

    def test_source_changes_and_replaced_lock_invalidate_candidate_approval(self):
        preview=self.prepare();self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError):self.apply(preview)
        self.path.write_bytes(self.original)
        lock=self.budget.lock_path;lock.rename(lock.with_name('old-lock.fixture'))
        lock.write_bytes(b'');lock.chmod(0o600)
        # A new coordinator must reject the prepared lock identity too.
        self.budget=PersistentBudget(self.root,scope=SCOPE,billing_scope=BILLING,policy=BudgetPolicy(max_attempts=8))
        with self.assertRaises(BudgetError):self.apply(preview)
        self.assert_original()

    def test_marker_blocks_prepare_apply_and_rollback(self):
        preview=self.prepare();self.budget.pending_path.touch(mode=0o600)
        for operation in (self.prepare,lambda:self.apply(preview),lambda:self.rollback(preview)):
            with self.assertRaises(BudgetError):operation()
        self.assert_original()

    def test_other_process_cannot_transition_while_lock_held(self):
        preview=self.prepare();ctx=multiprocessing.get_context('fork');recv,send=ctx.Pipe(False)
        with self.budget._locked():
            process=ctx.Process(target=child_apply,args=(self.root,preview.plan_sha256,send))
            process.start();self.assertTrue(recv.poll(5));self.assertEqual(recv.recv(),'blocked')
            process.join(5);self.assertEqual(process.exitcode,0)
        recv.close();send.close();self.assert_original()

    def test_pricing_is_revalidated_at_apply_not_only_prepare(self):
        preview=self.prepare()
        self.budget.clock=lambda:datetime.now(timezone.utc)+timedelta(days=1)
        with self.assertRaises(BudgetError):self.apply(preview)
        self.assert_original()

    def test_duplicate_prepare_apply_and_apply_after_rollback_are_rejected(self):
        preview=self.prepare()
        with self.assertRaises(BudgetError):self.prepare()
        self.apply(preview)
        with self.assertRaises(BudgetError):self.apply(preview)
        self.rollback(preview)
        with self.assertRaises(BudgetError):self.apply(preview)
        self.assert_original()

    def test_failed_atomic_write_retains_original_and_durable_recovery_guard(self):
        preview=self.prepare()
        with patch('northstar_atlas.live_budget.os.replace',side_effect=OSError('synthetic')):
            with self.assertRaises(BudgetError):self.apply(preview)
        self.assert_original()
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError):self.budget.inspect_history()
        with self.assertRaises(BudgetError):self.rollback(preview)

    def test_failed_apply_start_receipt_globally_blocks_new_transition_plans(self):
        preview=self.prepare();write=self.t._write_private
        def fail_start(budget,path,raw):
            if path.name=='apply.started':raise OSError('synthetic receipt failure')
            write(budget,path,raw)
        with patch.object(self.t,'_write_private',side_effect=fail_start):
            with self.assertRaises(BudgetError):self.apply(preview)
        self.assert_original()
        self.assertTrue(self.budget.pending_path.exists())
        self.price=self.price|{'id':'different-reviewed-price'}
        with self.assertRaises(BudgetError):self.prepare()

    def test_failed_rollback_start_receipt_keeps_v2_and_global_guard(self):
        preview=self.prepare();self.apply(preview);before=self.path.read_bytes()
        write=self.t._write_private
        def fail_start(budget,path,raw):
            if path.name=='rollback.started':raise OSError('synthetic receipt failure')
            write(budget,path,raw)
        with patch.object(self.t,'_write_private',side_effect=fail_start):
            with self.assertRaises(BudgetError):self.rollback(preview)
        self.assertEqual(self.path.read_bytes(),before)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError):self.budget.check_ready()

    def test_readback_failure_keeps_v2_with_guard_and_never_auto_restores(self):
        preview=self.prepare();write=self.budget._write
        def corrupt_after_write(value):
            write(value);self.path.write_bytes(b'{}')
        with patch.object(self.budget,'_write',side_effect=corrupt_after_write):
            with self.assertRaises(BudgetError):self.apply(preview)
        self.assertEqual(self.path.read_bytes(),b'{}')
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual((preview.artifact_directory/'backup.v1.json').read_bytes(),self.original)

    def test_process_crash_after_replace_blocks_new_writers_on_restart(self):
        preview=self.prepare();ctx=multiprocessing.get_context('fork')
        process=ctx.Process(target=child_apply,args=(self.root,preview.plan_sha256),kwargs={'crash':True})
        process.start();process.join(10);self.assertEqual(process.exitcode,0)
        self.assertEqual(json.loads(self.path.read_bytes())['version'],2)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError):self.budget.check_ready()
        with self.assertRaises(BudgetError):self.apply(preview)

    def test_rollback_requires_distinct_approval_and_restores_exact_original_bytes(self):
        preview=self.prepare();self.apply(preview)
        with self.assertRaises(BudgetError):self.t.rollback_transition(self.budget,preview.plan_sha256,
            approved_rollback_sha256=preview.plan_sha256,writers_stopped=True)
        self.rollback(preview);self.assert_original()
        self.assertEqual(self.budget.inspect_history().attempts_consumed,1)
        with self.assertRaises(BudgetError):self.budget.check_ready()

    def test_any_v2_reservation_even_zero_call_closed_workflow_prohibits_rollback(self):
        preview=self.prepare();self.apply(preview)
        with self.budget.reserve(WorkflowRequest('no-refund',LINEAR,'Who?','approved_only','chroma',date.today())):pass
        before=self.path.read_bytes()
        with self.assertRaises(BudgetError):self.rollback(preview)
        self.assertEqual(self.path.read_bytes(),before)

    def test_rollback_can_precede_pricing_expiry_but_cannot_erase_config_changes(self):
        preview=self.prepare();self.apply(preview)
        current=self.path.read_bytes()
        self.path.write_bytes(current+b' ')
        with self.assertRaises(BudgetError):self.rollback(preview)
        self.path.write_bytes(current)
        self.budget.clock=lambda:datetime.now(timezone.utc)+timedelta(days=1)
        self.rollback(preview);self.assert_original()

    def test_packet_symlinks_permissions_and_path_traversal_rejected(self):
        preview=self.prepare();backup=preview.artifact_directory/'backup.v1.json'
        backup.chmod(0o644)
        with self.assertRaises(BudgetError):self.apply(preview)
        backup.chmod(0o600);saved=backup.with_suffix('.fixture');backup.rename(saved);backup.symlink_to(saved)
        with self.assertRaises(BudgetError):self.apply(preview)
        for plan in ('../elsewhere','1'*63,'not-a-digest'):
            with self.assertRaises(BudgetError):self.t.apply_transition(self.budget,plan,approved_plan_sha256=plan,writers_stopped=True)
        self.assert_original()

    def test_cli_prepare_apply_and_rollback_only_use_fixture_paths(self):
        price=self.root/'.local/verified-price.json';price.write_bytes(encoded(self.price));price.chmod(0o600)
        common=[sys.executable,'-B','-m','northstar_atlas.budget_transition','--root',str(self.root)]
        args=['prepare','--pricing-file','.local/verified-price.json','--verified-pricing-sha256',digest(encoded(self.price)),
            '--confirm-billing-scope',BILLING,'--approve-max-attempts','8','--approve-max-reserved-usd','0.100000','--all-writers-stopped']
        def run(arguments):
            proc=subprocess.run(common+arguments,env=os.environ.copy(),capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stderr)
            return json.loads(proc.stdout)
        preview=run(args+['--dry-run']);self.assert_original()
        run(args)
        run(['apply','--plan',preview['plan_sha256'],'--approve',preview['plan_sha256'],'--all-writers-stopped'])
        self.assertEqual(self.budget.check_ready().version,2)
        run(['rollback','--plan',preview['plan_sha256'],'--approve',digest(('rollback:'+preview['plan_sha256']).encode()),'--all-writers-stopped'])
        self.assert_original()

    def test_cli_errors_are_safe_and_do_not_echo_private_ledger_fields(self):
        self.path.write_bytes(b'{"sensitive-fixture-value":"do not echo"}')
        proc=subprocess.run([sys.executable,'-B','-m','northstar_atlas.budget_transition','--root',str(self.root),
            'apply','--plan','a'*64,'--approve','a'*64,'--all-writers-stopped'],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0)
        self.assertNotIn('sensitive-fixture-value',proc.stdout+proc.stderr)
        self.assertNotIn('Traceback',proc.stdout+proc.stderr)
