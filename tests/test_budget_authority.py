"""Separate checkout and ledger authority; synthetic files, no provider traffic."""
from datetime import date, timedelta
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import socket
import unittest
from unittest.mock import patch

from budget_fixture import BILLING, SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from northstar_atlas.live_budget import (
    BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH, LINEAR, configured_budget,
)


def external_process(app, authority, send, *, crash=False):
    """A second checkout/process must contend for the very same POSIX lock."""
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(app)
    budget = PersistentBudget(app, authority_root=authority, authorized_authority_root=authority,
        scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(max_attempts=8))
    try:
        if crash:
            with budget.reserve(WorkflowRequest('crashed', GRAPH, 'Who?', 'approved_only', 'chroma', date.today())):
                os._exit(0)
        budget.check_ready()
    except BudgetError:
        send.send('blocked')
    else:
        send.send('ready')


class BudgetAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.app = checkout(self)
        self.authority = checkout(self)
        self.path = install(self.authority, document(calls=8))
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.app),
            NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
            NORTHSTAR_BUDGET_MAX_ATTEMPTS='8', NORTHSTAR_CONTROLLED_LIVE='0',
            NORTHSTAR_BUDGET_AUTHORITY_ROOT=str(self.authority),
            NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED=str(self.authority)))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))

    def test_authorized_environment_uses_external_ledger_without_redirecting_application(self):
        budget = configured_budget(self.app)
        self.assertEqual(budget.root, self.app)
        self.assertEqual(budget.path, self.path)
        self.assertEqual(budget.check_ready().attempts_consumed, 0)
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def budget(self):
        return PersistentBudget(self.app, authority_root=self.authority,
            authorized_authority_root=self.authority, scope=SCOPE, billing_scope=BILLING,
            policy=BudgetPolicy(max_attempts=8))

    def spec(self, ident='one', profile=LINEAR):
        return WorkflowRequest(ident, profile, 'Who?', 'approved_only', 'chroma', date.today())

    def test_same_root_default_and_explicit_same_root_keep_existing_behavior(self):
        local = install(self.app, document(calls=8))
        with patch.dict(os.environ):
            os.environ.pop('NORTHSTAR_BUDGET_AUTHORITY_ROOT')
            os.environ.pop('NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED')
            budget = configured_budget(self.app)
            self.assertEqual(budget.path, local)
            self.assertEqual(budget.check_ready().version, 2)
            os.environ['NORTHSTAR_BUDGET_AUTHORITY_ROOT'] = str(self.app)
            self.assertEqual(configured_budget(self.app).check_ready().version, 2)

    def test_external_configuration_requires_exact_confirmation_before_ledger_io(self):
        cases = [(str(self.authority), None), (str(self.authority), '1'),
                 (str(self.authority), str(self.app)), ('', ''),
                 (None, str(self.authority)), (str(self.authority), '')]
        before = self.path.read_bytes()
        for selected, confirmed in cases:
            with self.subTest(selected=selected, confirmed=confirmed), patch.dict(os.environ):
                for name, value in [('NORTHSTAR_BUDGET_AUTHORITY_ROOT', selected),
                                    ('NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED', confirmed)]:
                    if value is None: os.environ.pop(name, None)
                    else: os.environ[name] = value
                with patch.object(PersistentBudget, '_read', side_effect=AssertionError('ledger read')):
                    with self.assertRaises(BudgetError): configured_budget(self.app).check_ready()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_constructor_also_rejects_unapproved_authority(self):
        with self.assertRaises(BudgetError):
            PersistentBudget(self.app, authority_root=self.authority)

    def test_canonical_paths_reject_aliases_traversal_relative_and_missing_roots(self):
        alias = self.app / 'authority-link'; alias.symlink_to(self.authority, target_is_directory=True)
        paths = [alias, str(self.authority / '..' / self.authority.name),
                 '.local/elsewhere', str(self.authority) + '/./', self.app / 'missing']
        for path in paths:
            with self.subTest(path=path), self.assertRaises(BudgetError):
                PersistentBudget(self.app, authority_root=path, authorized_authority_root=path)

    def test_symlinked_ledger_and_lock_are_rejected_without_touching_targets(self):
        for name in ('budget.json', 'budget.lock'):
            path = self.path.with_name(name); saved = path.with_name(name + '.fixture')
            path.rename(saved); original = saved.read_bytes(); path.symlink_to(saved)
            try:
                with self.assertRaises(BudgetError): self.budget().check_ready()
                self.assertEqual(saved.read_bytes(), original)
            finally:
                path.unlink(); saved.rename(path)

    def test_symlinked_storage_directories_are_rejected(self):
        for path in (self.authority / '.local', self.path.parent):
            saved = path.with_name(path.name + '-fixture')
            path.rename(saved); path.symlink_to(saved, target_is_directory=True)
            try:
                with self.assertRaises(BudgetError): self.budget().check_ready()
            finally:
                path.unlink(); saved.rename(path)

    def test_restrictive_permissions_cover_authority_ancestors_and_files(self):
        for path, mode in ((self.authority, 0o777), (self.authority / '.local', 0o775),
                           (self.path.parent, 0o750), (self.path, 0o644),
                           (self.path.with_name('budget.lock'), 0o640)):
            saved = path.stat().st_mode & 0o7777
            path.chmod(mode)
            try:
                with self.subTest(path=path), self.assertRaises(BudgetError): self.budget().check_ready()
            finally: path.chmod(saved)

    def test_wrong_owner_on_any_authority_component_fails_closed(self):
        original = Path.lstat
        for target in (self.authority, self.authority / '.local', self.path.parent,
                       self.path, self.path.with_name('budget.lock')):
            def wrong_owner(path, *args, **kwargs):
                info = original(path, *args, **kwargs)
                if path == target:
                    values = list(info); values[4] = os.geteuid() + 1
                    return os.stat_result(values)
                return info
            with self.subTest(target=target), patch.object(Path, 'lstat', wrong_owner):
                with self.assertRaises(BudgetError): self.budget().check_ready()

    def test_nonregular_and_hardlinked_ledger_files_fail_before_read(self):
        saved = self.path.with_name('original.fixture'); self.path.rename(saved)
        try:
            os.mkfifo(self.path, 0o600)
            with self.assertRaises(BudgetError): self.budget().check_ready()
            self.path.unlink(); os.link(saved, self.path)
            with self.assertRaises(BudgetError): self.budget().check_ready()
        finally:
            self.path.unlink(); saved.rename(self.path)

    def test_missing_or_corrupt_external_ledger_never_falls_back_to_local(self):
        local = install(self.app, document(calls=8)); local_bytes = local.read_bytes()
        self.path.write_text('{}')
        with self.assertRaises(BudgetError): self.budget().check_ready()
        self.path.unlink()
        with self.assertRaises(BudgetError):
            with self.budget().reserve(self.spec()): pass
        self.assertFalse(self.path.exists()); self.assertEqual(local.read_bytes(), local_bytes)

    def test_missing_lock_is_not_created(self):
        lock = self.path.with_name('budget.lock'); lock.unlink()
        with self.assertRaises(BudgetError): self.budget().check_ready()
        self.assertFalse(lock.exists())

    def test_expired_unpriced_and_incompatible_ledgers_fail_without_rewrite(self):
        for change in ('expired', 'unpriced', 'incompatible'):
            value = document(calls=8)
            if change == 'expired':
                from datetime import datetime, timezone
                value['pricing_revisions'][0]['valid_until'] = (datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
            elif change == 'unpriced': value['pricing_revisions'] = []
            else: value['billing_scope'] = 'different-fixture-scope'
            self.path.write_text(json.dumps(value)); before = self.path.read_bytes()
            with self.subTest(change=change), self.assertRaises(BudgetError): self.budget().check_ready()
            self.assertEqual(self.path.read_bytes(), before)

    def test_environment_reconfiguration_cannot_redirect_cached_coordinator(self):
        budget = configured_budget(self.app); budget.check_ready()
        before = self.path.read_bytes()
        with patch.dict(os.environ, NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED=''):
            with self.assertRaises(BudgetError):
                with budget.reserve(self.spec()): pass
        self.assertEqual(self.path.read_bytes(), before)
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_application_hashes_and_invocation_binding_still_use_application_root(self):
        # The authority is storage, not a second source of policy truth.
        (self.authority / 'data/corpus_manifest.json').write_text('{}')
        budget = self.budget()
        spec = self.spec()
        with budget.reserve(spec): pass
        manifest_hash = hashlib.sha256((self.app / 'data/corpus_manifest.json').read_bytes()).hexdigest()
        self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['binding'], spec.binding(manifest_hash))
        before = self.path.read_bytes()
        (self.app / 'data/policies/HR-001.md').write_text('corrupt synthetic fixture')
        with self.assertRaises(ValueError):
            with budget.reserve(self.spec('changed')): pass
        self.assertEqual(self.path.read_bytes(), before)

    def test_replaced_lock_cannot_split_cached_coordinator_lock_identity(self):
        budget = self.budget(); budget.check_ready()
        lock = self.path.with_name('budget.lock'); lock.rename(lock.with_name('retired.fixture'))
        lock.write_text(''); lock.chmod(0o600)
        with self.assertRaises(BudgetError): budget.check_ready()

    def test_two_checkout_processes_share_the_authoritative_lock(self):
        ctx = multiprocessing.get_context('fork'); recv, send = ctx.Pipe(False)
        with self.budget().reserve(self.spec()):
            process = ctx.Process(target=external_process, args=(self.authority, self.authority, send))
            process.start()
            try:
                self.assertTrue(recv.poll(5)); self.assertEqual(recv.recv(), 'blocked')
            finally: process.join(5); recv.close(); send.close()
            self.assertEqual(process.exitcode, 0)

    def test_crashed_external_workflow_survives_restart_and_blocks_duplicate(self):
        ctx = multiprocessing.get_context('fork')
        process = ctx.Process(target=external_process, args=(self.app, self.authority, None), kwargs={'crash': True})
        process.start(); process.join(10); self.assertEqual(process.exitcode, 0)
        restarted = self.budget()
        self.assertEqual(restarted.inspect_history().attempts_consumed, 7)
        with self.assertRaises(BudgetError):
            with restarted.reserve(self.spec('crashed', GRAPH)): pass
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_graph_and_linear_share_limits_and_duplicate_ids_across_checkouts(self):
        budget = self.budget(); provider = Provider()
        with budget.reserve(self.spec('graph', GRAPH)) as reservation:
            self.assertEqual(reservation.envelope, (7, 184064, 20736))
            for stage, ordinal, output, context in [('router',0,256,8192),('grader',1,4096,32768),
                ('grader',2,4096,32768),('generator',0,2048,32768),('semantic_checker',1,4096,32768),
                ('generator',1,2048,32768),('semantic_checker',2,4096,32768)]:
                reservation.invoke(provider, stage, ordinal, request(output, context))
        with patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.authority)):
            same = PersistentBudget(self.authority, scope=SCOPE, billing_scope=BILLING,
                                    policy=BudgetPolicy(max_attempts=8))
            with self.assertRaises(BudgetError):
                with same.reserve(self.spec('graph')): pass
            with same.reserve(self.spec('linear')) as reservation:
                reservation.invoke(provider, 'generator', 0, request())
        self.assertEqual(budget.inspect_history().attempts_consumed, 8)
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec('overdraw')): pass
        self.assertEqual(len(provider.calls), 8)
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_timeout_keeps_external_full_workflow_reservation(self):
        budget = self.budget()
        with budget.reserve(self.spec(profile=GRAPH)) as reservation:
            with self.assertRaises(TimeoutError):
                reservation.invoke(Provider(TimeoutError('synthetic')), 'router', 0, request(256,8192))
        state = self.budget().inspect_history()
        self.assertEqual(state.attempts_consumed, 7)
        self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['status'], 'uncertain')

    def test_external_version_one_history_remains_read_only(self):
        price = document()['pricing_revisions'][0].copy(); price.pop('id')
        from northstar_atlas.llm import MODEL_ID
        value = dict(version=1, scope=SCOPE, model=MODEL_ID, max_attempts=3,
                     max_reserved_usd='0.100000', pricing=price, attempts=[])
        self.path.write_text(json.dumps(value)); before = self.path.read_bytes()
        self.assertEqual(self.budget().inspect_history().version, 1)
        with self.assertRaises(BudgetError): self.budget().check_ready()
        self.assertEqual(self.path.read_bytes(), before)

    def test_dotenv_cannot_select_budget_authority(self):
        (self.app / '.env').write_text('NORTHSTAR_BUDGET_AUTHORITY_ROOT=' + str(self.authority) + '\n'
            + 'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED=' + str(self.authority) + '\n')
        with patch.dict(os.environ):
            for name in ('NORTHSTAR_BUDGET_AUTHORITY_ROOT', 'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED'):
                os.environ.pop(name)
            budget = configured_budget(self.app)
            self.assertEqual(budget.path, self.app / '.local/controlled-live/budget.json')
            with self.assertRaises(BudgetError): budget.check_ready()
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_failed_external_settlement_blocks_restart_without_local_fallback(self):
        budget = self.budget(); write = budget._write
        def fail_closed_write(value):
            if value['workflows'][-1]['status'] == 'closed': raise OSError('synthetic storage failure')
            write(value)
        with patch.object(budget, '_write', side_effect=fail_closed_write):
            with self.assertRaises(BudgetError):
                with budget.reserve(self.spec()) as reservation:
                    reservation.invoke(Provider(), 'generator', 0, request())
        self.assertTrue(self.path.with_name('settlement.pending').is_file())
        with self.assertRaises(BudgetError): self.budget().check_ready()
        self.assertFalse((self.app / '.local/controlled-live').exists())

    def test_real_linear_service_uses_external_reservation_without_contract_changes(self):
        from test_qa import FakeLLM, FixtureStore, authored_claims
        from test_context import AS_OF, chunk, hit
        from northstar_atlas.generation import prepare_answer
        from northstar_atlas.qa import QAService
        hits = (hit(chunk()),)
        provider = FakeLLM(authored_claims(prepare_answer('Who?', hits, mode='draft_review', reference_date=AS_OF)))
        service = QAService(FixtureStore(hits), None, provider, corpus_fingerprint='fixture')
        with self.budget().reserve(WorkflowRequest('qa', LINEAR, 'Who?', 'draft_review', 'custom', AS_OF)) as reservation:
            result = service.ask('Who?', mode='draft_review', reference_date=AS_OF, reservation=reservation)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.status, 'not_assessed')
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(self.budget().inspect_history().attempts_consumed, 1)
        self.assertFalse((self.app / '.local/controlled-live').exists())
