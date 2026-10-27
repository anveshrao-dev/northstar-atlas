"""Synthetic cooperating-writer protocol tests; no live ledger or provider.

The small legacy probe models the reviewed lock/schema boundary, not the full
legacy accounting implementation. Separate source-review evidence is documented
in Budget-Ledger-Transition.md.
"""
from contextlib import redirect_stdout
from dataclasses import replace
import fcntl
import io
import json
import multiprocessing
import os
import unittest
from unittest.mock import patch

import test_budget_transition as existing
from test_budget_transition import digest, encoded
from budget_fixture import BILLING, SCOPE
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget


def legacy_probe(root, send):
    """Restarted v1 protocol double: lock and schema check precede fake inference."""
    folder = root / '.local/controlled-live'
    try:
        with (folder/'budget.lock').open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            value = json.loads((folder/'budget.json').read_bytes())
            if type(value['version']) is not int or value['version'] != 1:
                send.send('schema-rejected')
                return
            send.send('fake-inference')
    except BlockingIOError:
        send.send('lock-rejected')


def child_transition(root, plan, approval, send, crash):
    from northstar_atlas import budget_transition as t
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING,
                              policy=BudgetPolicy(max_attempts=8))
    if crash:
        write = budget._write
        def die(value):
            if crash == 'after':
                write(value)
            os._exit(0)
        budget._write = die
    try:
        t.apply_transition(budget, plan, approved_plan_sha256=plan,
                           writers_stopped=False, cooperating_writers=approval)
        send.send('applied')
    except BudgetError:
        send.send('blocked')


class CooperatingTests(unittest.TestCase):
    def setUp(self):
        existing.TransitionTests.setUp(self)
        self.assertTrue(hasattr(self.t, 'CooperatingWriters'), 'explicit cooperating-writer mode absent')
        lock = self.budget.lock_path.stat()
        directory = self.budget.directory.stat()
        self.provenance = dict(version=1, kind='reviewed-v1-lock-cooperation',
            authority_root=str(self.root), directory_identity=[directory.st_dev, directory.st_ino],
            lock_identity=[lock.st_dev, lock.st_ino],
            writers=[dict(writer_id='synthetic-legacy-writer',
                ledger_source_sha256='1'*64, provider_source_sha256='2'*64)],
            guarantees=dict(all_deployed_writers_reviewed=True, same_ledger_and_lock=True,
                exclusive_lock_through_inference_and_settlement=True,
                fresh_v1_schema_check_before_inference=True,
                no_cached_authorization_or_unlocked_writes=True,
                no_lock_or_directory_replacement=True))
        self.provenance_path = self.root/'.local/reviewed-writers.json'
        self.set_provenance(self.provenance)

    def set_provenance(self, value):
        raw = encoded(value)
        self.provenance_path.write_bytes(raw)
        self.provenance_path.chmod(0o600)
        self.approval = self.t.CooperatingWriters(self.provenance_path, digest(raw), True)

    def prepare(self, **changes):
        options = dict(writers_stopped=False, cooperating_writers=self.approval)
        options.update(changes)
        return existing.TransitionTests.prepare(self, **options)

    def apply(self, preview, **changes):
        options = dict(writers_stopped=False, cooperating_writers=self.approval)
        options.update(changes)
        return self.t.apply_transition(self.budget, preview.plan_sha256,
            approved_plan_sha256=preview.plan_sha256, **options)

    def rollback(self, preview):
        return existing.TransitionTests.rollback(self, preview)

    assert_original = existing.TransitionTests.assert_original

    def probe(self):
        ctx = multiprocessing.get_context('fork')
        receive, send = ctx.Pipe(False)
        process = ctx.Process(target=legacy_probe, args=(self.root, send))
        try:
            process.start()
            self.assertTrue(receive.poll(5))
            result = receive.recv()
            process.join(5)
            self.assertEqual(process.exitcode, 0)
            return result
        finally:
            if process.is_alive():
                process.terminate(); process.join(5)
            receive.close(); send.close()

    def test_explicit_mode_acknowledgment_and_review_digest_required(self):
        for approval in (None, replace(self.approval, acknowledged=False),
                         replace(self.approval, approved_sha256='0'*64)):
            with self.subTest(approval=approval), self.assertRaises(BudgetError):
                self.prepare(cooperating_writers=approval)
        with self.assertRaises(BudgetError):
            self.prepare(writers_stopped=True)
        self.assert_original()
        self.assertEqual({p.name for p in self.path.parent.iterdir()}, {'budget.json','budget.lock'})

    def test_dry_run_and_history_conservation(self):
        before = set(self.path.parent.iterdir())
        dry = self.prepare(dry_run=True)
        self.assertEqual(set(self.path.parent.iterdir()), before)
        preview = self.prepare()
        self.assertEqual(dry.plan_sha256, preview.plan_sha256)
        folder = preview.artifact_directory
        self.assertEqual((folder/'backup.v1.json').read_bytes(), self.original)
        self.assertEqual((folder/'writer-provenance.json').read_bytes(), self.provenance_path.read_bytes())
        self.apply(preview)
        self.assertEqual(json.loads(self.path.read_bytes())['legacy'], self.legacy)
        self.assertEqual(self.budget.check_ready().attempts_consumed, 1)
        self.assertEqual(str(self.budget.check_ready().reserved_usd), '0.000130')
        self.assertEqual(len(list(self.path.parent.glob('budget.json'))), 1)

    def test_apply_needs_repeated_acknowledgment_and_same_provenance(self):
        preview = self.prepare()
        with self.assertRaises(BudgetError):
            self.apply(preview, cooperating_writers=replace(self.approval, acknowledged=False))
        self.set_provenance(self.provenance | {'writers':[self.provenance['writers'][0] | {'writer_id':'changed-review'}]})
        with self.assertRaises(BudgetError):
            self.apply(preview)
        self.assert_original()

    def test_modes_cannot_silently_reinterpret_each_others_plans(self):
        cooperating = self.prepare()
        with self.assertRaises(BudgetError):
            self.apply(cooperating, writers_stopped=True, cooperating_writers=None)
        stopped = self.prepare(writers_stopped=True, cooperating_writers=None)
        with self.assertRaises(BudgetError):
            self.apply(stopped)
        self.assertNotEqual(cooperating.plan_sha256, stopped.plan_sha256)
        self.assert_original()

    def test_malformed_unreviewed_or_conflicting_provenance_rejected(self):
        writer = self.provenance['writers'][0]
        cases = [self.provenance | {'version':True}, self.provenance | {'kind':'unreviewed'},
            self.provenance | {'authority_root':str(self.root/'other')},
            self.provenance | {'writers':[]}, self.provenance | {'writers':[writer, writer]},
            self.provenance | {'writers':[writer | {'ledger_source_sha256':'unknown'}]},
            self.provenance | {'directory_identity':[True, 1]},
            self.provenance | {'lock_identity':[1, 2]}, self.provenance | {'extra':True}]
        cases += [self.provenance | {'guarantees':self.provenance['guarantees'] | {key:False}}
                  for key in self.provenance['guarantees']]
        for value in cases:
            self.set_provenance(value)
            with self.subTest(value=value), self.assertRaises(BudgetError):
                self.prepare()
        self.assert_original()

    def test_provenance_permissions_symlink_duplicates_and_tampering_rejected(self):
        self.provenance_path.chmod(0o644)
        with self.assertRaises(BudgetError): self.prepare()
        self.provenance_path.chmod(0o600)
        raw = self.provenance_path.read_bytes()
        self.provenance_path.write_bytes(raw+b' ')
        with self.assertRaises(BudgetError): self.prepare()
        self.provenance_path.write_bytes(b'{"version":1,"version":1}')
        self.approval = replace(self.approval, approved_sha256=digest(self.provenance_path.read_bytes()))
        with self.assertRaises(BudgetError): self.prepare()
        self.set_provenance(self.provenance)
        moved = self.provenance_path.with_suffix('.original')
        self.provenance_path.rename(moved); self.provenance_path.symlink_to(moved)
        with self.assertRaises(BudgetError): self.prepare()

    def test_packet_provenance_cannot_be_tampered_with(self):
        preview = self.prepare()
        (preview.artifact_directory/'writer-provenance.json').write_bytes(b'{}')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assert_original()

    def test_original_bytes_change_invalidates_candidate(self):
        preview = self.prepare()
        self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original+b' ')
        self.assertFalse((preview.artifact_directory/'apply.started').exists())

    def test_replaced_lock_rejected_even_by_new_coordinator(self):
        preview = self.prepare()
        lock = self.budget.lock_path
        lock.rename(lock.with_suffix('.old')); lock.touch(mode=0o600)
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                      policy=BudgetPolicy(max_attempts=8))
        with self.assertRaises(BudgetError): self.apply(preview)
        with self.assertRaises(BudgetError): self.prepare()
        self.assert_original()

    def test_replaced_authority_directory_rejected_with_same_lock_inode(self):
        preview = self.prepare()
        folder = self.budget.directory
        old = folder.with_name('old-fixture'); folder.rename(old); folder.mkdir(mode=0o700)
        for path in old.iterdir(): path.rename(folder/path.name)
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                      policy=BudgetPolicy(max_attempts=8))
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assert_original()

    def test_restarted_legacy_writer_denied_through_write_fsync_and_readback(self):
        preview = self.prepare()
        write = self.budget._write
        sync = self.budget._sync_directory
        read = self.budget._read
        def observe(value):
            self.assertEqual(self.probe(), 'lock-rejected')
            write(value)
            self.assertEqual(self.probe(), 'lock-rejected')
        def observe_sync():
            self.assertEqual(self.probe(), 'lock-rejected')
            sync()
        def observe_read():
            self.assertEqual(self.probe(), 'lock-rejected')
            return read()
        with patch.object(self.budget, '_write', side_effect=observe), \
             patch.object(self.budget, '_sync_directory', side_effect=observe_sync), \
             patch.object(self.budget, '_read', side_effect=observe_read):
            self.apply(preview)
        self.assertEqual(self.probe(), 'schema-rejected')

    def test_legacy_holding_lock_denies_preparation_and_application(self):
        preview = self.prepare()
        with self.budget.lock_path.open('r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BudgetError): self.prepare()
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assert_original()

    def test_failed_write_preserves_bytes_and_blocks_transition_recovery(self):
        preview = self.prepare()
        with patch('northstar_atlas.live_budget.os.replace', side_effect=OSError('synthetic failure')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assert_original()
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview)
        with self.assertRaises(BudgetError): self.rollback(preview)
        # Legacy has no settlement-marker check: unchanged v1 remains its authority.
        self.assertEqual(self.probe(), 'fake-inference')

    def test_crash_after_replace_denies_both_old_and_new_writers(self):
        preview = self.prepare()
        ctx = multiprocessing.get_context('fork'); receive, send = ctx.Pipe(False)
        process = ctx.Process(target=child_transition,
            args=(self.root, preview.plan_sha256, self.approval, send, 'after'))
        process.start(); process.join(10)
        self.assertEqual(process.exitcode, 0)
        receive.close(); send.close()
        self.assertEqual(self.probe(), 'schema-rejected')
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.budget.check_ready()
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_crash_before_replace_keeps_v1_and_requires_manual_recovery(self):
        preview = self.prepare()
        ctx = multiprocessing.get_context('fork'); receive, send = ctx.Pipe(False)
        process = ctx.Process(target=child_transition,
            args=(self.root, preview.plan_sha256, self.approval, send, 'before'))
        process.start(); process.join(10)
        self.assertEqual(process.exitcode, 0)
        receive.close(); send.close()
        self.assert_original()
        self.assertEqual(self.probe(), 'fake-inference')
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview)
        with self.assertRaises(BudgetError): self.rollback(preview)

    def test_change_during_preparation_is_rejected_not_returned_for_approval(self):
        write = self.t._write_private
        def changed(budget, path, raw):
            write(budget, path, raw)
            if path.name == 'plan.json':
                self.path.write_bytes(self.original+b' ')
        with patch.object(self.t, '_write_private', side_effect=changed):
            with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(), self.original+b' ')

    def test_change_after_validation_before_settlement_is_not_overwritten(self):
        preview = self.prepare()
        settle = self.budget._settle
        def changed(value, **kwargs):
            self.path.write_bytes(self.original+b' ')
            settle(value, **kwargs)
        with patch.object(self.budget, '_settle', side_effect=changed):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original+b' ')
        self.assertTrue(self.budget.pending_path.exists())

    def test_pending_history_and_settlement_markers_still_block(self):
        pending = self.legacy | {'attempts':[self.legacy['attempts'][0] |
                                          {'status':'reserved','finished_at':None}]}
        self.path.write_bytes(encoded(pending))
        with self.assertRaises(BudgetError): self.prepare()
        self.path.write_bytes(self.original)
        self.budget.pending_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.prepare()
        self.assert_original()

    def test_expired_pricing_and_wrong_billing_still_block(self):
        with self.assertRaises(BudgetError): self.prepare(confirmed_billing_scope='different')
        preview = self.prepare()
        from datetime import datetime, timedelta, timezone
        self.budget.clock = lambda:datetime.now(timezone.utc)+timedelta(days=1)
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assert_original()

    def test_missing_ledger_does_not_initialize_a_second_one(self):
        self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual({p.name for p in self.path.parent.iterdir()}, {'budget.lock'})

    def test_readback_failure_blocks_v2_and_never_restores_v1(self):
        preview = self.prepare()
        with patch.object(self.budget, '_read', side_effect=BudgetError('synthetic readback failure')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(json.loads(self.path.read_bytes())['version'], 2)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.probe(), 'schema-rejected')
        with self.assertRaises(BudgetError): self.rollback(preview)

    def test_rollback_still_requires_stopped_writers_and_exact_unused_candidate(self):
        preview = self.prepare(); self.apply(preview)
        with self.assertRaises(BudgetError):
            self.t.rollback_transition(self.budget, preview.plan_sha256,
                approved_rollback_sha256=digest(('rollback:'+preview.plan_sha256).encode()),
                writers_stopped=False)
        self.rollback(preview)
        self.assert_original()
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_cli_explicit_mode_no_implicit_approval(self):
        price = self.root/'.local/price.json'; price.write_bytes(encoded(self.price)); price.chmod(0o600)
        base = ['--root', str(self.root), 'prepare', '--pricing-file','.local/price.json',
            '--verified-pricing-sha256',digest(encoded(self.price)), '--confirm-billing-scope',BILLING,
            '--approve-max-attempts','8','--approve-max-reserved-usd','0.100000',
            '--cooperating-writers', '--writer-provenance','.local/reviewed-writers.json',
            '--approve-writer-provenance',self.approval.approved_sha256]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(self.t.main(base+['--dry-run']), 1)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.t.main(base+['--acknowledge-cooperating-writers-only']), 0)
        preview = json.loads(output.getvalue())
        args = ['--root',str(self.root),'apply','--plan',preview['plan_sha256'],
            '--approve',preview['plan_sha256'],'--cooperating-writers',
            '--writer-provenance','.local/reviewed-writers.json',
            '--approve-writer-provenance',self.approval.approved_sha256,
            '--acknowledge-cooperating-writers-only']
        with redirect_stdout(io.StringIO()): self.assertEqual(self.t.main(args), 0)
        self.assertEqual(self.budget.check_ready().version, 2)
