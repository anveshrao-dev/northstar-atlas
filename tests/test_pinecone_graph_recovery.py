"""Synthetic recovery records only; no real authority, credentials or network."""
from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import importlib
import importlib.util
import json
import io
import multiprocessing
import os
import socket
import unittest
from unittest.mock import patch

from budget_fixture import BILLING, SCOPE, document
import test_linear_demo as demo_fixture
import test_live_graph as graph_fixture
import test_orchestration_semantic as semantic_fixture
from test_context import chunk, hit, AS_OF
from test_bm25 import catalog
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas import diagnostic_window as window
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, GRAPH


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def child_apply(root, plan, send, crash=False):
    from northstar_atlas import pinecone_graph_recovery as recovery
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(15))
    if crash:
        write = budget._write
        def interrupted(value):
            write(value)
            os._exit(0)
        budget._write = interrupted
    try:
        recovery.apply_recovery(budget, plan, approved_plan_sha256=plan,
            approved_rebind_sha256=recovery.rebind_approval(plan), writers_stopped=True, recover_identity=True)
    except BudgetError:
        send.send('blocked')
    else:
        send.send('applied')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.pinecone_graph_recovery'),
                             'approved recovery implementation is missing')
        self.r = importlib.import_module('northstar_atlas.pinecone_graph_recovery')
        fixture = demo_fixture.LinearDemoTests(methodName='runTest')
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.root, self.path, self.budget = fixture.root, fixture.path, fixture.budget
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        # Authored historical device-change fixture: same inode, different old device.
        self.window = self.path.with_name(window.WINDOW_NAME)
        old = json.loads(self.window.read_bytes())
        for key in ('directory_identity', 'lock_identity'):
            old[key][0] += 1
        self.window.write_bytes(encode(old))
        self.old_window = self.window.read_bytes()
        self.old_source = self.path.read_bytes()
        self.history = json.loads(self.old_source)
        self.previous = window.sha(self.old_window)
        self.spec = replace(fixture.spec, invocation_id='one-pinecone-graph',
                            mode='approved_only', profile=GRAPH)
        # Copy only the public frozen snapshot, never any real ledger/publication identity.
        from checkout_fixture import ROOT
        from northstar_atlas.live_graph import SNAPSHOT_ID
        relative = f'.local/pinecone/snapshots/{SNAPSHOT_ID}.json'
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
        self.contract = self.r.request_contract(self.budget, self.spec)
        observed = self.r.inspect_recovery(self.budget, writers_stopped=True, recover_identity=True)
        self.continuity = dict(version=1, authority=str(self.budget.authority_root), scope=SCOPE,
            billing_scope=BILLING, source_sha256=window.sha(self.old_source),
            previous_window_sha256=self.previous, identities_sha256=observed['identities_sha256'],
            independent_evidence_sha256=window.sha(b'synthetic independently reviewed continuity record'))
        self.price = document()['pricing_revisions'][0]
        self.price['id'] = 'fresh-recovery-price'

    def prepare(self, **changes):
        options = dict(approved_continuity_sha256=window.sha(encode(self.continuity)),
            approved_max_attempts=22, approved_max_reserved_usd='0.100000',
            pricing_revision=self.price, verified_pricing_sha256=window.sha(encode(self.price)),
            writers_stopped=True, recover_identity=True, dry_run=False)
        options.update(changes)
        return self.r.prepare_recovery(self.budget, self.contract, self.continuity, **options)

    def apply(self, preview, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256,
            approved_rebind_sha256=self.r.rebind_approval(preview.plan_sha256),
            writers_stopped=True, recover_identity=True)
        options.update(changes)
        return self.r.apply_recovery(self.budget, preview.plan_sha256, **options)

    def updated(self):
        return PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
            policy=BudgetPolicy(22), before_request=getattr(self, 'access', None))

    def activate(self):
        self.preview = self.prepare(); self.apply(self.preview)
        self.access = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.budget = self.updated()
        self.enterContext(patch.dict(os.environ, NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
            NORTHSTAR_UI_ENABLE_GRAPH='1', NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1', NORTHSTAR_UI_ENABLE_PINECONE='1'))

    def wire(self, **options):
        # Reuse the real Nebius serializer, durable stage permits and semantic graph
        # with only the HTTP/model/retrieval boundaries replaced by authored fixtures.
        f = graph_fixture.LiveGraphTests(methodName='runTest')
        self.addCleanup(f.doCleanups)
        f.root, f.path, f.budget = self.root, self.path, self.budget
        f.created, f.sent, f.closed = [], [], []
        f.fixture = semantic_fixture.PhaseBSemanticGraphTests(methodName='runTest')
        f.fixture.setUp()
        f.fixture.source = chunk(doc='HR-001', status=APPROVED, effective=AS_OF)
        f.fixture.cat = catalog([f.fixture.source]); f.fixture.hits = (hit(f.fixture.source),)
        f.wire(backend='pinecone', **options)
        return f

    def run_once(self, **changes):
        options = dict(plan_sha256=self.preview.plan_sha256,
            approved_run_sha256=self.r.run_approval(self.preview.plan_sha256),
            question_sha256=window.sha(self.spec.question.encode()), allow_live=True, other_writers_stopped=True)
        options.update(changes)
        return self.r.run_recovery(self.budget, self.spec, **options)

    def test_old_identity_stays_rejected_and_dry_run_does_not_write(self):
        with self.assertRaises(BudgetError):
            self.budget.inspect_history(diagnostic_approval=self.previous)
        before = {p: p.read_bytes() for p in self.path.parent.rglob('*') if p.is_file()}
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(before, {p: p.read_bytes() for p in self.path.parent.rglob('*') if p.is_file()})
        self.assertEqual(preview.attempts_preserved, 15)
        with self.assertRaises(BudgetError): self.budget.check_ready()

    def test_explicit_recovery_and_independent_continuity_approval_required(self):
        for changes in (dict(writers_stopped=False), dict(recover_identity=False),
                        dict(approved_continuity_sha256='0'*64), dict(verified_pricing_sha256='0'*64),
                        dict(approved_max_attempts=16), dict(approved_max_attempts=23),
                        dict(approved_max_reserved_usd='0.2')):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.prepare(**changes)
        for key in ('source_sha256','previous_window_sha256','identities_sha256','billing_scope'):
            original = self.continuity[key]; self.continuity[key] = '0'*64
            with self.subTest(key=key), self.assertRaises(BudgetError): self.prepare()
            self.continuity[key] = original
        self.assertEqual(self.path.read_bytes(), self.old_source)

    def test_exact_candidate_preserves_history_and_byte_exact_backups(self):
        preview = self.prepare()
        self.assertEqual((preview.artifact_directory/'backup.v2.json').read_bytes(), self.old_source)
        self.assertEqual((preview.artifact_directory/'backup.window.json').read_bytes(), self.old_window)
        for p in preview.artifact_directory.iterdir(): self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertEqual(preview.artifact_directory.stat().st_mode & 0o777, 0o700)
        self.apply(preview)
        actual = json.loads(self.path.read_bytes())
        expected = deepcopy(self.history)
        expected.update(max_attempts=22, active_pricing=self.price['id'],
                        diagnostic_request_sha256=window.sha(encode(self.contract)))
        expected['pricing_revisions'].append(self.price)
        self.assertEqual(actual, expected)
        state = self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)
        self.assertEqual(state.attempts_consumed, 15)
        self.assertEqual(state.reserved_usd, Decimal(preview.reserved_usd))
        self.assertEqual(actual['max_reserved_usd'], '0.100000')
        self.assertEqual((preview.artifact_directory/'backup.window.json').read_bytes(), self.old_window)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_candidate_apply_requires_separate_exact_plan_and_rebind_approval(self):
        preview = self.prepare()
        for changes in (dict(approved_plan_sha256='0'*64), dict(approved_rebind_sha256='0'*64),
                        dict(writers_stopped=False), dict(recover_identity=False)):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.apply(preview, **changes)
        self.assertEqual(self.path.read_bytes(), self.old_source)
        candidate = preview.artifact_directory/'candidate.v2.json'
        candidate.write_bytes(candidate.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_source_byte_change_invalidates_candidate(self):
        preview = self.prepare(); self.path.write_bytes(self.old_source+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.window.read_bytes(), self.old_window)

    def test_current_lock_replacement_invalidates_candidate(self):
        preview = self.prepare()
        replacement = self.path.with_name('new.lock'); replacement.write_bytes(b''); replacement.chmod(0o600)
        os.replace(replacement, self.budget.lock_path)
        with self.assertRaises(BudgetError): self.apply(preview)
        # Also reject after a new process captures the replacement identity.
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(15))
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_expired_or_unaffordable_pricing_rejected_at_prepare_and_apply(self):
        high = {**self.price, 'input_per_million_usd':'100', 'output_per_million_usd':'100'}
        with self.assertRaises(BudgetError):
            self.prepare(pricing_revision=high, verified_pricing_sha256=window.sha(encode(high)))
        preview = self.prepare(); now = self.budget.clock()
        self.budget.clock = lambda: now+timedelta(days=2)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_failed_atomic_write_leaves_guard_and_no_live_authorization(self):
        preview = self.prepare()
        with patch.object(self.budget, '_write', side_effect=OSError('private persistence detail')):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), self.old_source)
        with self.assertRaises(BudgetError):
            self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_missing_ledger_is_never_initialized(self):
        self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.path.exists())

    def test_current_identity_change_rejected_after_install_and_restart(self):
        preview = self.prepare(); self.apply(preview)
        real_identity = window.identity
        def changed(path):
            pair = real_identity(path)
            return [pair[0]+1,pair[1]] if path == self.budget.directory else pair
        with patch.object(window, 'identity', side_effect=changed):
            with self.assertRaises(BudgetError):
                self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_only_exact_approved_only_pinecone_graph_can_reserve_once(self):
        preview = self.prepare(); self.apply(preview); budget = self.updated()
        for request in (replace(self.spec, backend='chroma'), replace(self.spec, mode='draft_review'),
                        replace(self.spec, invocation_id='other'), replace(self.spec, question=self.spec.question+' '),
                        replace(self.spec, reference_date=self.spec.reference_date+timedelta(days=1))):
            with self.subTest(request=request), self.assertRaises(BudgetError):
                with budget.reserve(request, diagnostic_approval=preview.plan_sha256): pass
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec): pass
        with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256) as permit:
            self.assertEqual(permit.envelope, (7,184064,20736))
            value = json.loads(self.path.read_bytes())
            self.assertEqual(value['workflows'][-1]['backend'], 'pinecone')
            self.assertEqual(value['workflows'][-1]['status'], 'reserved')
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256): pass
        with self.assertRaises(BudgetError): budget.check_ready()

    def test_one_seven_stage_graph_is_reserved_before_io_then_duplicate_denied(self):
        self.activate(); f = self.wire(judgments=('uncertain','relevant'), semantics=('incomplete','supported'))
        result = self.run_once()
        self.assertTrue(result.accepted)
        self.assertEqual(result.kind, 'answer')
        self.assertEqual(result.public.coverage.status, 'not_assessed')
        self.assertIn('(approved, synthetic)', result.public.text)
        self.assertEqual(f.sent, ['router','grader','grader','generator','semantic_checker','generator','semantic_checker'])
        self.assertEqual(len(f.created), 1); self.assertEqual(len(f.closed), 1)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['legacy'], self.history['legacy'])
        self.assertEqual(value['workflows'][:2], self.history['workflows'])
        self.assertEqual(value['workflows'][2]['status'], 'closed')
        self.assertFalse(self.run_once().accepted)
        self.assertEqual(len(f.sent), 7)

    def test_separate_live_approval_and_gates_are_required_before_transport(self):
        self.activate(); f = self.wire()
        for changes in (dict(allow_live=False),dict(other_writers_stopped=False),
                        dict(approved_run_sha256='0'*64),dict(question_sha256='0'*64)):
            with self.subTest(changes=changes): self.assertFalse(self.run_once(**changes).accepted)
        self.assertFalse(f.created); self.assertEqual(len(json.loads(self.path.read_bytes())['workflows']), 2)
        with patch.dict(os.environ, NORTHSTAR_UI_ENABLE_PINECONE='0'):
            self.assertFalse(self.run_once().accepted)
        self.assertFalse(f.sent)

    def test_router_timeout_preserves_whole_reservation_and_safe_diagnostics(self):
        import httpx
        self.activate(); f = self.wire(failure=httpx.ReadTimeout('PRIVATE-CREDENTIAL-PAYLOAD'))
        result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        value = json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'][:2], self.history['workflows'])
        self.assertEqual(value['workflows'][-1]['status'], 'uncertain')
        self.assertEqual(self.budget.inspect_history(diagnostic_approval=self.preview.plan_sha256).attempts_consumed,22)
        diagnostic = self.path.with_name('provider-failure.json').read_text()
        self.assertIn('transport_timeout', diagnostic); self.assertNotIn('PRIVATE', diagnostic)
        self.assertFalse(self.run_once().accepted); self.assertEqual(f.sent, ['router'])

    def test_remote_audit_failure_withholds_inference_and_blocks_reuse(self):
        self.activate(); f = self.wire()
        with patch('northstar_atlas.orchestration.open_retriever', side_effect=ValueError('PRIVATE audit failure')):
            result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        self.assertFalse(f.sent); self.assertEqual(len(f.closed), 1)
        self.assertFalse(self.run_once().accepted)

    def test_private_preview_failure_prevents_provider_construction(self):
        self.activate(); f = self.wire()
        self.access.side_effect = ValueError('private access detail')
        self.assertFalse(self.run_once().accepted)
        self.assertFalse(f.created); self.assertFalse(f.sent)

    def test_frozen_snapshot_tampering_prevents_reservation(self):
        self.activate(); f = self.wire()
        snapshot = self.root / f".local/pinecone/snapshots/{self.contract['snapshot_id']}.json"
        snapshot.write_bytes(snapshot.read_bytes()+b' ')
        self.assertFalse(self.run_once().accepted)
        self.assertFalse(f.created); self.assertEqual(len(json.loads(self.path.read_bytes())['workflows']),2)

    def test_pending_corrupt_and_wrong_history_fail_without_writes(self):
        for mutate in (lambda v:v.update(max_attempts=16), lambda v:v.update(max_reserved_usd='0.2'),
                       lambda v:v['workflows'].pop(), lambda v:v.update(legacy=None),
                       lambda v:v['workflows'][0].update(status='reserved',finished_at=None)):
            value=deepcopy(self.history); mutate(value); self.path.write_bytes(encode(value))
            with self.assertRaises(BudgetError): self.prepare()
        self.path.write_bytes(self.old_source)
        self.budget.pending_path.write_bytes(b'')
        with self.assertRaises(BudgetError): self.prepare()

    def test_wrong_snapshot_mode_and_historical_ids_rejected_at_preparation(self):
        for key, bad in (('snapshot_id','0'*64),('snapshot_sha256','0'*64),('model','wrong-model'),
                         ('backend','chroma'),('profile','linear_v1'),('mode','draft_review'),
                         ('invocation_id',self.history['legacy']['attempts'][0]['id']),
                         ('invocation_id',self.history['workflows'][0]['id'])):
            saved=self.contract[key]; self.contract[key]=bad
            with self.subTest(key=key,bad=bad),self.assertRaises(BudgetError): self.prepare()
            self.contract[key]=saved

    def test_artifact_symlink_and_unrelated_window_rejected(self):
        preview=self.prepare()
        candidate=preview.artifact_directory/'candidate.v2.json'
        saved=candidate.with_name('saved.json'); candidate.rename(saved); candidate.symlink_to(saved)
        with self.assertRaises(BudgetError): self.apply(preview)
        candidate.unlink(); saved.rename(candidate)
        self.window.write_bytes(self.old_window+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_lock_contention_across_processes_does_not_replace_lock(self):
        preview=self.prepare(); before=window.identity(self.budget.lock_path)
        ctx=multiprocessing.get_context('spawn'); receiver,sender=ctx.Pipe(False)
        child=ctx.Process(target=child_apply,args=(self.root,preview.plan_sha256,sender))
        with self.budget._physical_lock():
            child.start(); child.join(10)
            if child.is_alive(): child.terminate(); child.join(); self.fail('child hung on lock')
            self.assertEqual(child.exitcode,0); self.assertTrue(receiver.poll(2))
            self.assertEqual(receiver.recv(),'blocked')
        receiver.close(); sender.close()
        self.assertEqual(window.identity(self.budget.lock_path),before)
        self.assertEqual(self.path.read_bytes(),self.old_source)

    def test_crash_after_ledger_replace_keeps_guard_and_complete_history(self):
        preview=self.prepare(); ctx=multiprocessing.get_context('spawn'); receiver,sender=ctx.Pipe(False)
        child=ctx.Process(target=child_apply,args=(self.root,preview.plan_sha256,sender,True))
        child.start(); child.join(10)
        if child.is_alive(): child.terminate(); child.join(); self.fail('child hung')
        receiver.close(); sender.close(); self.assertEqual(child.exitcode,0)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(),(preview.artifact_directory/'candidate.v2.json').read_bytes())
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'],self.history['workflows'])
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_cli_inspect_is_safe_and_prepare_defaults_to_read_only(self):
        self.assertTrue(callable(getattr(self.r,'main',None)), 'recovery CLI is missing')
        files=[]
        for name,value in (('contract',self.contract),('continuity',self.continuity),('price',self.price)):
            p=self.root/'.local'/f'{name}.json'; p.write_bytes(encode(value)); p.chmod(0o600); files.append(p)
        common=['--root',str(self.root)]
        with patch.object(self.r,'configured_budget',return_value=self.budget),redirect_stdout(io.StringIO()) as output:
            self.assertEqual(self.r.main(common+['inspect','--writers-stopped','--recover-identity']),0)
            args=['prepare','--contract',str(files[0]),'--continuity',str(files[1]),
                '--approve-continuity',window.sha(encode(self.continuity)),
                '--approve-max-attempts','22','--approve-max-reserved-usd','0.100000',
                '--pricing-revision',str(files[2]),'--verified-pricing-sha256',window.sha(encode(self.price)),
                '--writers-stopped','--recover-identity']
            self.assertEqual(self.r.main(common+args),0)
        self.assertNotIn(SCOPE,output.getvalue()); self.assertNotIn(BILLING,output.getvalue())
        self.assertNotIn(str(self.root),output.getvalue())
        self.assertEqual(self.path.read_bytes(),self.old_source)
        self.assertFalse(list(self.path.parent.glob('pinecone-graph-recovery-*')))

    def test_parent_identity_change_invalidates_candidate_and_installed_approval(self):
        preview = self.prepare()
        original = window.identity
        for parent in (self.budget.authority_root, self.budget.authority_root/'.local'):
            def changed(path):
                pair = original(path)
                return [pair[0]+1, pair[1]] if path == parent else pair
            with self.subTest(parent=parent.name), patch.object(window, 'identity', side_effect=changed):
                with self.assertRaises(BudgetError): self.apply(preview)
        self.apply(preview)
        with patch.object(window, 'identity', side_effect=changed):
            with self.assertRaises(BudgetError):
                self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_unsafe_permissions_ownership_and_hardlinks_are_rejected(self):
        for path in (self.path, self.window, self.budget.lock_path):
            path.chmod(0o644)
            with self.subTest(path=path.name), self.assertRaises(BudgetError): self.prepare()
            path.chmod(0o600)
        with patch('os.geteuid', return_value=os.geteuid()+1):
            with self.assertRaises(BudgetError): self.prepare()
        link = self.path.with_name('hardlink.json'); os.link(self.path, link)
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(), self.old_source)

    def test_missing_or_malformed_independent_evidence_is_not_assumed(self):
        for value in (None, '', 'not-a-digest', True):
            self.continuity['independent_evidence_sha256'] = value
            with self.subTest(value=value), self.assertRaises(BudgetError): self.prepare()
        del self.continuity['independent_evidence_sha256']
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(), self.old_source)

    def test_window_replace_failure_retains_guard_and_originals(self):
        preview = self.prepare(); original = os.replace
        def fail_window(source, target):
            if target == self.window: raise OSError('private window replacement failure')
            return original(source, target)
        with patch.object(self.r.os, 'replace', side_effect=fail_window):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), self.old_source)
        self.assertEqual(self.window.read_bytes(), self.old_window)
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_window_fsync_failure_keeps_guard_before_ledger_replacement(self):
        preview = self.prepare(); original = self.r._sync
        def fail_window_sync(path):
            if path == preview.artifact_directory: raise OSError('private fsync failure')
            return original(path)
        with patch.object(self.r, '_sync', side_effect=fail_window_sync):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), self.old_source)
        self.assertEqual(window.sha(self.window.read_bytes()), preview.plan_sha256)
        with self.assertRaises(BudgetError): self.updated().check_ready()

    def test_readback_failure_keeps_guard_after_complete_candidate_write(self):
        preview = self.prepare(); original = PersistentBudget._read
        def fail_readback(budget, *args, **kwargs):
            if budget.policy.max_attempts == 22: raise BudgetError('private readback failure')
            return original(budget, *args, **kwargs)
        with patch.object(PersistentBudget, '_read', fail_readback):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(self.path.read_bytes(), (preview.artifact_directory/'candidate.v2.json').read_bytes())
        self.assertFalse((preview.artifact_directory/'apply.completed').exists())
        with self.assertRaises(BudgetError): self.updated().check_ready()

    def test_receipt_write_failure_does_not_release_guard_or_allow_retry(self):
        preview = self.prepare(); original = self.r._write_private
        def fail_receipt(budget, path, raw):
            if path.name == 'apply.completed': raise OSError('private receipt failure')
            return original(budget, path, raw)
        with patch.object(self.r, '_write_private', side_effect=fail_receipt):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'], self.history['workflows'])
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_expired_pricing_denies_run_before_provider_construction(self):
        self.activate(); f = self.wire(); now = self.budget.clock()
        self.budget.clock = lambda: now+timedelta(days=2)
        self.assertFalse(self.run_once().accepted)
        self.assertFalse(f.created); self.assertFalse(f.sent)
        self.assertEqual(len(json.loads(self.path.read_bytes())['workflows']), 2)

    def test_http_rejection_retains_full_reservation_without_retry(self):
        import httpx
        self.activate(); f = self.wire()
        with patch('httpx.Client.send', return_value=httpx.Response(400, text='PRIVATE-PAYLOAD',
                request=httpx.Request('POST', 'https://synthetic.invalid'))) as send:
            result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        self.assertEqual(send.call_count, 1)
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][-1]['status'], 'uncertain')
        raw = self.path.with_name('provider-failure.json').read_text()
        self.assertIn('http_rejection', raw); self.assertNotIn('PRIVATE', raw)
        self.assertFalse(self.run_once().accepted)
        self.assertEqual(len(f.created), 1)

    def test_malformed_router_envelope_never_reaches_retrieval_or_second_call(self):
        import httpx
        self.activate(); f = self.wire()
        with patch('httpx.Client.send', return_value=httpx.Response(200, text='PRIVATE-NOT-JSON',
                request=httpx.Request('POST', 'https://synthetic.invalid'))) as send:
            result = self.run_once()
        self.assertFalse(result.accepted); self.assertIsNone(result.public)
        self.assertEqual(send.call_count, 1)
        raw = self.path.with_name('provider-failure.json').read_text()
        self.assertIn('response_envelope_validation', raw); self.assertNotIn('PRIVATE', raw)
        flow = json.loads(self.path.read_bytes())['workflows'][-1]
        self.assertEqual((flow['status'], [c['stage'] for c in flow['calls']]), ('uncertain', ['router']))
        self.assertEqual(len(f.closed), 1)


if __name__ == '__main__':
    unittest.main()
