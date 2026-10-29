"""Dedicated demo amendments use synthetic ledgers and fake transports only."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from contextlib import redirect_stdout
import io
import json
import multiprocessing
import os
import unittest
from unittest.mock import patch

from budget_fixture import SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from northstar_atlas import demo_budget_amendment as amendment
from northstar_atlas import linear_demo
from northstar_atlas.diagnostic_window import sha
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, LINEAR, GRAPH

BILLING = 'northstar-atlas-week2-demo'


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def child_apply(root, plan, pipe):
    os.environ['NORTHSTAR_PROJECT_ROOT'] = str(root)
    budget = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(1))
    try:
        amendment.apply_amendment(budget, plan, approved_plan_sha256=plan, writers_stopped=True)
    except BudgetError:
        pipe.send('blocked')
    else:
        pipe.send('applied')


class DemoAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root),
            NORTHSTAR_CONTROLLED_LIVE='0'))
        initial = document(calls=1); initial['billing_scope'] = BILLING
        self.path = install(self.root, initial)
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(1))
        self.first = WorkflowRequest('first-demo', LINEAR, 'First fixture?', 'approved_only', 'pinecone', date(2026, 10, 29))
        with self.budget.reserve(self.first) as permit:
            permit.invoke(Provider(), 'generator', 0, request())
        self.original = self.path.read_bytes()
        self.history = json.loads(self.original)
        self.price = deepcopy(self.history['pricing_revisions'][0])
        self.spec = replace(self.first, invocation_id='second-demo', question='Second fixture?')
        self.contract = linear_demo.request_contract(self.budget, self.spec)

    def prepare(self, **changes):
        options = dict(confirmed_billing_scope=BILLING, approved_max_attempts=2,
            approved_max_reserved_usd='0.100000', verified_pricing_sha256=sha(encode(self.price)),
            writers_stopped=True, dry_run=False)
        options.update(changes)
        return amendment.prepare_amendment(self.budget, self.contract, **options)

    def apply(self, preview, **changes):
        options = dict(approved_plan_sha256=preview.plan_sha256, writers_stopped=True)
        options.update(changes)
        return amendment.apply_amendment(self.budget, preview.plan_sha256, **options)

    def updated(self, **kwargs):
        return PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(2), **kwargs)

    def test_dry_run_and_private_byte_exact_backup(self):
        before = {p.name: p.read_bytes() for p in self.path.parent.iterdir()}
        preview = self.prepare(dry_run=True)
        self.assertIsNone(preview.artifact_directory)
        self.assertEqual(preview.attempts_preserved, 1)
        self.assertEqual(Decimal(preview.reserved_usd), Decimal('0.003687'))
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.parent.iterdir()})
        saved = self.prepare()
        self.assertEqual(saved.plan_sha256, preview.plan_sha256)
        self.assertEqual((saved.artifact_directory/'backup.v2.json').read_bytes(), self.original)
        self.assertEqual(saved.artifact_directory.stat().st_mode & 0o777, 0o700)
        for p in saved.artifact_directory.iterdir(): self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_exact_candidate_and_history_conservation(self):
        preview = self.prepare(); self.apply(preview)
        expected = deepcopy(self.history)
        expected.update(max_attempts=2, diagnostic_request_sha256=sha(encode(self.contract)))
        self.assertEqual(json.loads(self.path.read_bytes()), expected)
        state = self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)
        self.assertEqual(state.attempts_consumed, 1)
        self.assertEqual(state.reserved_usd, Decimal('0.003687'))
        with self.assertRaises(BudgetError): self.apply(preview)
        with self.assertRaises(BudgetError): self.updated().check_ready()

    def test_exact_request_and_duplicate_enforcement(self):
        preview = self.prepare(); self.apply(preview)
        budget = self.updated()
        for spec in (replace(self.spec, invocation_id='other'), replace(self.spec, question='Other?'),
                     replace(self.spec, mode='draft_review'), replace(self.spec, backend='chroma'),
                     replace(self.spec, profile=GRAPH),
                     replace(self.spec, reference_date=self.spec.reference_date+timedelta(days=1))):
            with self.assertRaises(BudgetError):
                with budget.reserve(spec, diagnostic_approval=preview.plan_sha256): self.fail('unauthorized')
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec): self.fail('ordinary entrypoint')
        with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256) as permit:
            self.assertEqual(permit.envelope, (1, 30720, 2048))
            permit.invoke(Provider(), 'generator', 0, request())
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec, diagnostic_approval=preview.plan_sha256): self.fail('duplicate')
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][0], self.history['workflows'][0])

    def test_bad_approvals_contracts_and_duplicate_id_rejected(self):
        for changes in (dict(writers_stopped=False), dict(approved_max_attempts=3),
                        dict(approved_max_attempts=True), dict(approved_max_reserved_usd='0.20'),
                        dict(confirmed_billing_scope='other'), dict(verified_pricing_sha256='0'*64)):
            with self.subTest(changes=changes), self.assertRaises(BudgetError): self.prepare(**changes)
        original = deepcopy(self.contract)
        for field, value in [('mode','draft_review'), ('backend','chroma'), ('profile',GRAPH),
                             ('invocation_id','first-demo'), ('model','other'), ('corpus_sha256','0'*64)]:
            self.contract = {**original, field:value}
            with self.subTest(field=field), self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_wrong_source_history_or_billing_scope_rejected(self):
        for update in ({'max_attempts':2}, {'billing_scope':'other'}, {'workflows':[]},
                       {'version':1}, {'max_reserved_usd':'0.20'}, {'diagnostic_request_sha256':'a'*64}):
            self.path.write_bytes(encode({**self.history, **update}))
            with self.subTest(update=update), self.assertRaises(BudgetError): self.prepare()
        altered = deepcopy(self.history)
        altered['pricing_revisions'][0]['input_per_million_usd'] = '0.11'
        self.path.write_bytes(encode(altered))
        with self.assertRaises(BudgetError): self.prepare()

    def test_uncertain_first_call_conserved_not_refunded(self):
        value = deepcopy(self.history)
        value['workflows'][0]['status'] = 'uncertain'
        value['workflows'][0]['calls'][0].update(status='failed_or_uncertain', usage=None)
        self.path.write_bytes(encode(value))
        preview=self.prepare(); self.apply(preview)
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'], value['workflows'])
        self.assertEqual(Decimal(preview.reserved_usd), Decimal('0.003687'))

    def test_new_pricing_appended_without_repricing_history(self):
        fresh = {**self.price, 'id':'reviewed-new-price', 'output_per_million_usd':'0.4'}
        preview=self.prepare(pricing_revision=fresh, verified_pricing_sha256=sha(encode(fresh)))
        self.apply(preview)
        value=json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'], self.history['workflows'])
        self.assertEqual(value['pricing_revisions'][:-1], self.history['pricing_revisions'])
        self.assertLessEqual(Decimal(preview.workflow_reserved_usd), Decimal('0.01'))

    def test_stale_or_expensive_pricing_rejected(self):
        expensive={**self.price, 'id':'expensive', 'input_per_million_usd':'1'}
        with self.assertRaises(BudgetError):
            self.prepare(pricing_revision=expensive, verified_pricing_sha256=sha(encode(expensive)))
        with patch.object(self.budget, 'clock', return_value=self.budget.clock()+timedelta(days=2)):
            with self.assertRaises(BudgetError): self.prepare()
        preview=self.prepare()
        with patch.object(self.budget, 'clock', return_value=self.budget.clock()+timedelta(days=2)):
            with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_candidate_approval_and_source_bytes_binding(self):
        preview=self.prepare()
        with self.assertRaises(BudgetError): self.apply(preview, approved_plan_sha256='0'*64)
        self.path.write_bytes(self.original+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.path.write_bytes(self.original)
        candidate=preview.artifact_directory/'candidate.v2.json'
        candidate.write_bytes(candidate.read_bytes()+b' ')
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_filesystem_identity_change_rejected_even_after_restart(self):
        preview=self.prepare()
        self.budget.lock_path.rename(self.path.parent/'old.lock')
        self.budget.lock_path.touch(mode=0o600)
        self.budget=PersistentBudget(self.root,scope=SCOPE,billing_scope=BILLING,policy=BudgetPolicy(1))
        with self.assertRaises(BudgetError): self.apply(preview)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_lock_contention(self):
        preview=self.prepare(); ctx=multiprocessing.get_context('spawn'); parent, child=ctx.Pipe()
        with self.budget._locked():
            process=ctx.Process(target=child_apply,args=(self.root,preview.plan_sha256,child))
            process.start(); self.assertTrue(parent.poll(15)); self.assertEqual(parent.recv(),'blocked')
            process.join(15); self.assertEqual(process.exitcode,0)
        parent.close(); child.close()
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_persistence_failure_keeps_guard_and_no_automatic_restore(self):
        preview=self.prepare(); write=self.budget._write
        def uncertain(value):
            write(value)
            raise OSError('synthetic interruption')
        with patch.object(self.budget,'_write',side_effect=uncertain),self.assertRaises(BudgetError):
            self.apply(preview)
        self.assertTrue(self.budget.pending_path.exists())
        self.assertEqual(json.loads(self.path.read_bytes())['max_attempts'],2)
        self.assertEqual((preview.artifact_directory/'backup.v2.json').read_bytes(),self.original)
        with self.assertRaises(BudgetError): self.updated().inspect_history(diagnostic_approval=preview.plan_sha256)

    def test_missing_ledger_never_initialized(self):
        self.path.unlink()
        with self.assertRaises(BudgetError): self.prepare()
        self.assertFalse(self.path.exists())

    def test_existing_one_shot_runner_and_capture(self):
        from test_linear_demo import LinearDemoTests
        self.preview=self.prepare(); self.apply(self.preview)
        access=self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.budget=self.updated(before_request=access)
        self.enterContext(patch.dict(os.environ,NORTHSTAR_CONTROLLED_LIVE='1'))
        from test_context import chunk
        from northstar_atlas.ingestion.document_loader import APPROVED
        with patch('test_linear_demo.chunk', return_value=chunk(status=APPROVED, effective=date(2020, 1, 1))):
            LinearDemoTests.wire(self)
        def run(**changes):
            opts=dict(plan_sha256=self.preview.plan_sha256, approved_run_sha256=linear_demo.run_approval(self.preview.plan_sha256),
                question_sha256=sha(self.spec.question.encode()),allow_live=True,other_writers_stopped=True)
            opts.update(changes)
            return linear_demo.run_demo(self.budget,self.spec,**opts)
        self.assertFalse(run(allow_live=False).accepted); self.assertEqual(self.sent,[])
        outcome=run(); self.assertTrue(outcome.accepted)
        self.assertEqual(len(self.sent),1)
        self.assertFalse(run().accepted); self.assertEqual(len(self.sent),1)
        captured=json.loads((self.root/'.local/linear-demo-results/second-demo.json').read_bytes())
        self.assertEqual(captured['answer_text'],outcome.public.text)
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][0],self.history['workflows'][0])

    def test_cli_dry_run_only_prints_digests_counts(self):
        contract=self.root/'.local/contract.json'; contract.write_bytes(encode(self.contract)); contract.chmod(0o600)
        output=io.StringIO()
        with patch.object(amendment,'configured_budget',return_value=self.budget),redirect_stdout(output):
            code=amendment.main(['--root',str(self.root),'prepare','--contract',str(contract),
                '--confirmed-billing-scope',BILLING,'--approve-max-attempts','2',
                '--approve-max-reserved-usd','0.100000','--verified-pricing-sha256',sha(encode(self.price)),
                '--writers-stopped'])
        self.assertEqual(code,0); self.assertEqual(self.path.read_bytes(),self.original)
        self.assertNotIn(BILLING,output.getvalue()); self.assertNotIn(str(self.root),output.getvalue())


    def test_historical_graph_authority_is_rejected_and_untouched(self):
        from test_linear_demo import LinearDemoTests
        historical=LinearDemoTests(methodName='runTest')
        historical.setUp(); self.addCleanup(historical.doCleanups)
        original=historical.path.read_bytes()
        self.assertEqual(len(json.loads(original)['workflows']),2)
        self.assertTrue(all(f['status']=='uncertain' for f in json.loads(original)['workflows']))
        with self.assertRaises(BudgetError):
            amendment.prepare_amendment(historical.budget, self.contract,
                confirmed_billing_scope=BILLING,approved_max_attempts=2,
                approved_max_reserved_usd='0.100000',writers_stopped=True,
                verified_pricing_sha256=sha(encode(self.price)))
        with patch.dict(os.environ,NORTHSTAR_PROJECT_ROOT=str(self.root)):
            preview=self.prepare(); self.apply(preview)
        self.assertEqual(historical.path.read_bytes(),original)

    def test_pending_or_corrupt_ledger_and_unsafe_permissions_rejected(self):
        self.budget.pending_path.touch(mode=0o600)
        with self.assertRaises(BudgetError): self.prepare()
        self.budget.pending_path.unlink()
        self.path.write_bytes(b'{broken')
        with self.assertRaises(BudgetError): self.prepare()
        self.path.write_bytes(self.original); self.path.chmod(0o644)
        with self.assertRaises(BudgetError): self.prepare()

    def test_pending_first_writer_and_existing_window_rejected(self):
        value=deepcopy(self.history)
        value['workflows'][0].update(status='reserved',finished_at=None)
        self.path.write_bytes(encode(value))
        with self.assertRaises(BudgetError): self.prepare()
        self.path.write_bytes(self.original)
        window=self.path.with_name('diagnostic-window.json')
        window.write_bytes(b'{}'); window.chmod(0o600)
        with self.assertRaises(BudgetError): self.prepare()
        self.assertEqual(window.read_bytes(),b'{}')

    def test_runtime_identity_changes_and_expiry_fail_before_fake_provider(self):
        from northstar_atlas import diagnostic_window as window
        preview=self.prepare(); self.apply(preview)
        budget=self.updated(); provider=Provider()
        real_identity=window.identity
        def changed_device(path):
            device,inode=real_identity(path)
            return [device+1,inode]
        with patch.object(window,'identity',side_effect=changed_device),self.assertRaises(BudgetError):
            with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256) as permit:
                permit.invoke(provider,'generator',0,request())
        budget=self.updated(clock=lambda:self.budget.clock()+timedelta(days=2))
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256) as permit:
                permit.invoke(provider,'generator',0,request())
        self.assertEqual(provider.calls,[])

    def test_second_call_timeout_remains_reserved_without_retry(self):
        preview=self.prepare(); self.apply(preview)
        budget=self.updated(); provider=Provider(TimeoutError('synthetic'))
        with self.assertRaises(TimeoutError):
            with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256) as permit:
                permit.invoke(provider,'generator',0,request())
        self.assertEqual(len(provider.calls),1)
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256): self.fail('retry')
        value=json.loads(self.path.read_bytes())
        self.assertEqual(value['workflows'][0],self.history['workflows'][0])
        self.assertEqual(value['workflows'][1]['status'],'uncertain')
        state=budget.inspect_history(diagnostic_approval=preview.plan_sha256)
        self.assertEqual(state.attempts_consumed,2)
        self.assertEqual(state.reserved_usd,Decimal('0.007374'))

    def test_tampered_history_cannot_be_approved_by_a_recomputed_digest(self):
        # Even a matching plan/candidate digest must reconstruct the permitted changes.
        preview=self.prepare()
        candidate_path=preview.artifact_directory/'candidate.v2.json'
        candidate=json.loads(candidate_path.read_bytes())
        candidate['workflows'][0]['id']='rewritten-history'
        candidate_raw=encode(candidate); candidate_path.write_bytes(candidate_raw)
        plan_path=preview.artifact_directory/'plan.json'
        plan=json.loads(plan_path.read_bytes()); plan['candidate_sha256']=sha(candidate_raw)
        plan_raw=encode(plan); plan_path.write_bytes(plan_raw)
        next_id=sha(plan_raw)
        preview.artifact_directory.rename(preview.artifact_directory.with_name('demo-amendment-'+next_id))
        with self.assertRaises(BudgetError):
            amendment.apply_amendment(self.budget,next_id,approved_plan_sha256=next_id,writers_stopped=True)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_exact_cent_ceiling_accepts_boundary_rejects_rounding_above(self):
        price={**self.price,'id':'cent-boundary','input_per_million_usd':'0.3',
               'output_per_million_usd':'0.3828125'}
        preview=self.prepare(pricing_revision=price,verified_pricing_sha256=sha(encode(price)),dry_run=True)
        self.assertEqual(Decimal(preview.workflow_reserved_usd),Decimal('0.010000'))
        price['output_per_million_usd']='0.3828126'
        with self.assertRaises(BudgetError):
            self.prepare(pricing_revision=price,verified_pricing_sha256=sha(encode(price)),dry_run=True)
        self.assertEqual(self.path.read_bytes(),self.original)

    def test_apply_failure_before_replace_retains_original_and_guard(self):
        preview=self.prepare()
        with patch.object(self.budget,'_write',side_effect=OSError('synthetic')), self.assertRaises(BudgetError):
            self.apply(preview)
        self.assertEqual(self.path.read_bytes(),self.original)
        self.assertTrue(self.budget.pending_path.exists())
        with self.assertRaises(BudgetError): self.apply(preview)

    def test_empty_donor_from_real_transition_preserved_and_duplicate_rejected(self):
        from northstar_atlas import budget_transition as transition
        from test_budget_transition import history
        legacy=history(); legacy['attempts']=[]  # Valid expired historical pricing is retained.
        donor_raw=(json.dumps(legacy,indent=3)+'\n').encode()
        self.path.write_bytes(donor_raw)
        transition_preview=transition.prepare_transition(self.budget,self.price,
            confirmed_billing_scope=BILLING,approved_max_attempts=1,
            approved_max_reserved_usd='0.100000',verified_pricing_sha256=sha(encode(self.price)),
            writers_stopped=True,dry_run=False)
        transition.apply_transition(self.budget,transition_preview.plan_sha256,
            approved_plan_sha256=transition_preview.plan_sha256,writers_stopped=True)
        with self.budget.reserve(self.first) as permit:
            permit.invoke(Provider(),'generator',0,request())
        original=self.path.read_bytes(); original_value=json.loads(original)
        state_before=self.budget.inspect_history()
        preview=self.prepare()
        candidate_raw=(preview.artifact_directory/'candidate.v2.json').read_bytes()
        # The canonical donor representation produced by the existing transition
        # is preserved as identical bytes, including all historical price fields.
        donor_member=b'"legacy":'+encode(legacy)+b',"max_attempts":'
        self.assertIn(donor_member,original)
        self.assertIn(donor_member,candidate_raw)
        self.assertEqual((preview.artifact_directory/'backup.v2.json').read_bytes(),original)
        self.assertEqual((transition_preview.artifact_directory/'backup.v1.json').read_bytes(),donor_raw)
        self.apply(preview)
        self.assertIn(donor_member,self.path.read_bytes())
        candidate=json.loads(self.path.read_bytes())
        self.assertEqual(candidate['legacy'],legacy)
        self.assertEqual(candidate['workflows'],original_value['workflows'])
        self.assertEqual(candidate['pricing_revisions'],original_value['pricing_revisions'])
        self.assertEqual(candidate['max_attempts'],2)
        self.assertEqual(candidate['max_reserved_usd'],'0.100000')
        budget=self.updated()
        self.assertEqual(budget.inspect_history(diagnostic_approval=preview.plan_sha256),state_before)
        self.assertEqual(state_before.reserved_usd,Decimal('0.003687'))
        with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256) as permit:
            permit.invoke(Provider(),'generator',0,request())
        with self.assertRaises(BudgetError):
            with budget.reserve(self.spec,diagnostic_approval=preview.plan_sha256): self.fail('duplicate')
        self.assertIn(donor_member,self.path.read_bytes())
        self.assertEqual(json.loads(self.path.read_bytes())['workflows'][0],original_value['workflows'][0])

    def test_valid_empty_unpriced_legacy_donor_is_retained(self):
        from test_budget_transition import history
        legacy=history(); legacy.update(attempts=[],pricing=None)
        original={**self.history,'legacy':legacy}
        self.budget._validate_document(original,current=False)
        self.path.write_bytes(encode(original))
        preview=self.prepare(); self.apply(preview)
        self.assertEqual(json.loads(self.path.read_bytes())['legacy'],legacy)
        self.assertIn(b'"legacy":'+encode(legacy)+b',"max_attempts":',self.path.read_bytes())
        self.assertEqual(self.updated().inspect_history(diagnostic_approval=preview.plan_sha256).reserved_usd,
                         Decimal('0.003687'))

    def test_nonempty_malformed_and_incompatible_legacy_rejected(self):
        from test_budget_transition import history
        from northstar_atlas.live_budget import _legacy
        nonempty=history()
        self.assertEqual(_legacy(nonempty)[0],1)  # Structurally valid, but out of scope.
        empty={**nonempty,'attempts':[]}
        donors=[nonempty,{},[],False,
            {**empty,'version':2}, {**empty,'scope':'f'*32}, {**empty,'model':'other'},
            {**empty,'max_attempts':1}, {**empty,'max_reserved_usd':'0.20'},
            {**empty,'attempts':{}}, {**empty,'attempts':None},
            {**empty,'pricing':{}}, {**empty,'unexpected':'field'}]
        before=set(self.path.parent.iterdir())
        for donor in donors:
            raw=encode({**self.history,'legacy':donor}); self.path.write_bytes(raw)
            with self.subTest(donor=donor),self.assertRaises(BudgetError): self.prepare()
            self.assertEqual(self.path.read_bytes(),raw)
            self.assertEqual(set(self.path.parent.iterdir()),before)

    def test_recomputed_candidate_digest_cannot_rewrite_empty_legacy(self):
        from test_budget_transition import history
        legacy=history(); legacy['attempts']=[]
        original=encode({**self.history,'legacy':legacy}); self.path.write_bytes(original)
        preview=self.prepare()
        candidate_path=preview.artifact_directory/'candidate.v2.json'
        candidate=json.loads(candidate_path.read_bytes())
        candidate['legacy']['pricing']['source']='https://nebius.com/prices/changed'
        raw=encode(candidate); candidate_path.write_bytes(raw)
        plan_path=preview.artifact_directory/'plan.json'
        plan=json.loads(plan_path.read_bytes()); plan['candidate_sha256']=sha(raw)
        plan_raw=encode(plan); plan_path.write_bytes(plan_raw)
        plan_id=sha(plan_raw)
        preview.artifact_directory.rename(preview.artifact_directory.with_name('demo-amendment-'+plan_id))
        with self.assertRaises(BudgetError):
            amendment.apply_amendment(self.budget,plan_id,approved_plan_sha256=plan_id,writers_stopped=True)
        self.assertEqual(self.path.read_bytes(),original)
