"""Durable accounting with contained synthetic files and fake providers."""
import importlib.util
import json
import os
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import multiprocessing
from pathlib import Path
import unittest
from unittest.mock import patch

from checkout_fixture import checkout
from budget_fixture import BILLING, SCOPE, Provider, document, install, request


def crash_reserved(root):
    from northstar_atlas.live_budget import PersistentBudget, BudgetPolicy, WorkflowRequest
    b = PersistentBudget(root, scope=SCOPE, billing_scope=BILLING, policy=BudgetPolicy(max_attempts=8))
    with b.reserve(WorkflowRequest('crash', 'linear_v1', 'Who?', 'approved_only', 'chroma', date.today())):
        os._exit(0)


class Availability(unittest.TestCase):
    def test_shared_coordinator_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.live_budget'))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        if importlib.util.find_spec('northstar_atlas.live_budget') is None:
            self.skipTest('coordinator not implemented yet')
        from northstar_atlas import live_budget as b
        self.b = b
        self.root = checkout(self)
        self.env = patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root), NORTHSTAR_CONTROLLED_LIVE='0')
        self.env.start(); self.addCleanup(self.env.stop)
        self.path = install(self.root, document(calls=8))
        self.budget = b.PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
                                         policy=b.BudgetPolicy(max_attempts=8))

    def spec(self, ident='one', profile='linear_v1'):
        return self.b.WorkflowRequest(ident, profile, 'Who?', 'approved_only', 'chroma', date.today())

    def read(self):
        return json.loads(self.path.read_text())

    def test_all_seven_slots_are_reserved_before_any_send_then_shared_with_linear(self):
        p = Provider()
        with self.budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')) as r:
            self.assertEqual(r.envelope, (7, 184064, 20736))
            self.assertEqual(r.reserved_usd, Decimal('0.024629'))
            self.assertEqual(self.read()['workflows'][0]['status'], 'reserved')
            for stage, ordinal, output, context in [('router',0,256,8192),('grader',1,4096,32768),
                ('grader',2,4096,32768),('generator',0,2048,32768),('semantic_checker',1,4096,32768),
                ('generator',1,2048,32768),('semantic_checker',2,4096,32768)]:
                r.invoke(p, stage, ordinal, request(output, context))
        self.assertEqual(len(p.calls), 7)
        with self.budget.reserve(self.spec('linear')) as r:
            r.invoke(p, 'generator', 0, request())
        with self.assertRaises(self.b.BudgetError):
            with self.budget.reserve(self.spec('overdraw')): pass
        self.assertEqual(len(p.calls), 8)

    def test_default_policy_never_raises_three_call_allowance(self):
        original = document(calls=3); self.path.write_text(json.dumps(original))
        budget = self.b.PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING)
        with self.assertRaises(self.b.BudgetError):
            with budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')): pass
        self.assertEqual(self.read(), original)

    def test_duplicate_ids_and_changed_binding_never_replay(self):
        with self.budget.reserve(self.spec()): pass
        for spec in (self.spec(), replace(self.spec(), question='Different?')):
            with self.assertRaises(self.b.BudgetError):
                with self.budget.reserve(spec): pass
        self.assertEqual(len(self.read()['workflows']), 1)

    def test_order_duplicate_and_stage_token_caps_fail_before_provider(self):
        p=Provider()
        with self.budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')) as r:
            for stage, ordinal, req in [('generator',0,request()),('router',0,request(256,8192,8000))]:
                with self.assertRaises(self.b.BudgetError): r.invoke(p,stage,ordinal,req)
            r.invoke(p,'router',0,request(256,8192))
            with self.assertRaises(self.b.BudgetError): r.invoke(p,'router',0,request(256,8192))
            r.invoke(p,'grader',1,request(4096))
            with self.assertRaises(self.b.BudgetError): r.invoke(p,'generator',0,request(1024,32768,31000))
        self.assertEqual(len(p.calls),2)

    def test_timeout_holds_full_workflow_even_when_caller_catches_error(self):
        p=Provider(TimeoutError('synthetic'))
        with self.budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')) as r:
            with self.assertRaises(TimeoutError): r.invoke(p,'router',0,request(256,8192))
        self.assertEqual(self.read()['workflows'][0]['status'],'uncertain')
        ready=self.budget.check_ready()
        self.assertEqual(ready.attempts_consumed,7)
        self.assertEqual(ready.reserved_usd,Decimal('0.024629'))

    def test_unused_slots_released_but_usage_does_not_refund_started_stage(self):
        with self.budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')) as r:
            r.invoke(Provider(),'router',0,request(256,8192))
        state=self.budget.check_ready()
        self.assertEqual(state.attempts_consumed,1)
        self.assertEqual(state.reserved_usd,Decimal('0.000871'))

    def test_restart_keeps_unknown_reservation_and_blocks_duplicate(self):
        ctx=multiprocessing.get_context('fork')
        proc=ctx.Process(target=crash_reserved,args=(self.root,));proc.start();proc.join(10)
        self.assertEqual(proc.exitcode,0)
        self.assertEqual(self.budget.check_ready().attempts_consumed,1)
        with self.assertRaises(self.b.BudgetError):
            with self.budget.reserve(self.spec('crash')): pass

    def test_shared_lock_blocks_other_process(self):
        with self.budget.reserve(self.spec()):
            ctx=multiprocessing.get_context('fork'); recv,send=ctx.Pipe(False)
            def contender():
                try: self.budget.check_ready()
                except self.b.BudgetError: send.send('blocked')
                else: send.send('unsafe')
            proc=ctx.Process(target=contender);proc.start();self.assertTrue(recv.poll(5))
            self.assertEqual(recv.recv(),'blocked');proc.join(5)
            recv.close();send.close()

    def test_expired_corrupt_permissions_and_unknown_scope_fail_closed(self):
        original=self.path.read_bytes()
        cases=['{}','{"version":2,"version":2}']
        value=self.read();value['pricing_revisions'][0]['valid_until']=(datetime.now(timezone.utc)-timedelta(minutes=1)).isoformat()
        cases.append(json.dumps(value))
        for raw in cases:
            self.path.write_text(raw)
            with self.assertRaises(self.b.BudgetError): self.budget.check_ready()
        self.path.write_bytes(original);self.path.chmod(0o644)
        with self.assertRaises(self.b.BudgetError): self.budget.check_ready()
        self.path.chmod(0o600)
        with self.assertRaises(self.b.BudgetError):
            self.b.PersistentBudget(self.root,scope='2'*32,billing_scope=BILLING).check_ready()

    def test_failed_stage_persistence_sends_nothing(self):
        p=Provider()
        with self.assertRaises(self.b.BudgetError):
            with self.budget.reserve(self.spec()) as r:
                with patch.object(self.budget,'_write',side_effect=self.b.BudgetError('synthetic')):
                    r.invoke(p,'generator',0,request())
        self.assertFalse(p.calls)
        self.assertEqual(self.read()['workflows'][0]['status'],'reserved')

    def test_legacy_is_read_only_and_historical_attempts_still_count(self):
        price=document()['pricing_revisions'][0].copy();price.pop('id')
        legacy=dict(version=1,scope=SCOPE,model=self.b.MODEL_ID,max_attempts=3,max_reserved_usd='0.100000',
                    pricing=price,attempts=[dict(id='2'*32,reserved_at=price['confirmed_at'],finished_at=None,
                        input_tokens=1000,output_tokens=100,reserved_usd='0.000130',status='reserved')])
        raw=json.dumps(legacy);self.path.write_text(raw)
        self.assertEqual(self.budget.inspect_history().attempts_consumed,1)
        with self.assertRaises(self.b.BudgetError):
            with self.budget.reserve(self.spec()): pass
        self.assertEqual(self.path.read_text(),raw)
        self.path.write_text(json.dumps(document(calls=8,legacy=legacy)))
        with self.budget.reserve(self.spec(profile='llm_routing_grading_semantic_v1')): pass
        self.assertEqual(self.read()['legacy'],legacy)
        self.assertEqual(self.budget.check_ready().reserved_usd,Decimal('0.000130'))

    def test_missing_usage_is_uncertain_and_stops_further_calls(self):
        class Unknown(Provider):
            def generate(inner, req):
                return replace(super().generate(req), usage=None)
        with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
            with self.assertRaises(self.b.BudgetError):
                r.invoke(Unknown(),'router',0,request(256,8192))
            with self.assertRaises(self.b.BudgetError):
                r.invoke(Provider(),'grader',1,request(4096))
        self.assertEqual(self.budget.check_ready().attempts_consumed,7)
        self.assertEqual(self.read()['workflows'][0]['status'],'uncertain')

    def test_expiry_between_stages_sends_no_next_request(self):
        p=Provider()
        with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
            r.invoke(p,'router',0,request(256,8192))
            self.budget.clock=lambda: datetime.now(timezone.utc)+timedelta(days=1)
            with self.assertRaises(self.b.BudgetError):r.invoke(p,'grader',1,request(4096))
        self.assertEqual(len(p.calls),1)
        self.assertEqual(self.budget.inspect_history().attempts_consumed,1)

    def test_persistence_failure_after_response_never_releases_reservation(self):
        p=Provider();write=self.budget._write
        def fail_after_response(value):
            write(value)
            if value['workflows'][0]['calls'] and value['workflows'][0]['calls'][0]['status']=='returned':
                raise self.b.BudgetError('synthetic post-replacement fsync failure')
        with self.assertRaises(self.b.BudgetError):
            with patch.object(self.budget,'_write',side_effect=fail_after_response):
                with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
                    r.invoke(p,'router',0,request(256,8192))
        self.assertEqual(len(p.calls),1)
        self.assertEqual(self.read()['workflows'][0]['status'],'reserved')
        self.assertEqual(self.budget.check_ready().attempts_consumed,7)

    def test_corrupt_returned_record_without_usage_is_not_settled_history(self):
        with self.budget.reserve(self.spec()) as r:
            r.invoke(Provider(),'generator',0,request())
        value=self.read();value['workflows'][0]['calls'][0]['usage']=None
        self.path.write_text(json.dumps(value))
        with self.assertRaises(self.b.BudgetError):self.budget.check_ready()

    def test_no_ledger_creation_and_no_symlink_following(self):
        self.path.unlink()
        with self.assertRaises(self.b.BudgetError):self.budget.check_ready()
        self.assertFalse(self.path.exists())
        target=self.root/'.local/fixture.json';target.write_text(json.dumps(document(calls=8)))
        self.path.symlink_to(target)
        before=target.read_bytes()
        with self.assertRaises(self.b.BudgetError):
            with self.budget.reserve(self.spec()):pass
        self.assertEqual(target.read_bytes(),before)

    def test_post_replace_settlement_failure_blocks_restart_authorization(self):
        write=self.budget._write
        def ambiguous_settlement(value):
            write(value)
            if value['workflows'][-1]['status']=='closed':
                raise self.b.BudgetError('synthetic directory fsync failure after replacement')
        with patch.object(self.budget,'_write',side_effect=ambiguous_settlement):
            with self.assertRaises(self.b.BudgetError):
                with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
                    r.invoke(Provider(),'router',0,request(256,8192))
        restarted=self.b.PersistentBudget(self.root,scope=SCOPE,billing_scope=BILLING,
                                         policy=self.b.BudgetPolicy(max_attempts=8))
        with self.assertRaises(self.b.BudgetError):restarted.check_ready()
        with self.assertRaises(self.b.BudgetError):
            with restarted.reserve(self.spec('next')):pass

    def test_interrupted_settlement_guard_cleanup_blocks_restart(self):
        unlink=Path.unlink
        def interrupted(path,*args,**kwargs):
            if path.name=='settlement.pending':raise OSError('synthetic cleanup failure')
            return unlink(path,*args,**kwargs)
        with patch.object(Path,'unlink',interrupted):
            with self.assertRaises(self.b.BudgetError):
                with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
                    r.invoke(Provider(),'router',0,request(256,8192))
        with self.assertRaises(self.b.BudgetError):self.budget.check_ready()

    def test_cleanup_directory_sync_failure_restores_blocking_guard(self):
        sync=self.budget._sync_directory
        def fail_after_unlink():
            value=self.read()
            if (value['workflows'] and value['workflows'][-1]['status']=='closed'
                    and not self.budget.pending_path.exists()):
                raise OSError('synthetic guard removal sync failure')
            sync()
        with patch.object(self.budget,'_sync_directory',side_effect=fail_after_unlink):
            with self.assertRaises(self.b.BudgetError):
                with self.budget.reserve(self.spec(profile=self.b.GRAPH)) as r:
                    r.invoke(Provider(),'router',0,request(256,8192))
        self.assertTrue(self.budget.pending_path.is_file())
        restarted=self.b.PersistentBudget(self.root,scope=SCOPE,billing_scope=BILLING,
                                         policy=self.b.BudgetPolicy(max_attempts=8))
        with self.assertRaises(self.b.BudgetError):restarted.check_ready()
