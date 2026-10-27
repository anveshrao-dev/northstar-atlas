"""Real services, synthetic ledgers and mocked HTTP; no paid/provider requests."""
from contextlib import contextmanager
from datetime import date
import json
import os
import socket
import unittest
from unittest.mock import patch

import httpx
from budget_fixture import BILLING, SCOPE, Provider, document, install, request
from checkout_fixture import checkout
from northstar_atlas.live_budget import BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH, LINEAR
from northstar_atlas.llm import MODEL_ID


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.root=checkout(self)
        p=patch.dict(os.environ,NORTHSTAR_PROJECT_ROOT=str(self.root),NORTHSTAR_CONTROLLED_LIVE='0')
        p.start();self.addCleanup(p.stop)
        p=patch.object(socket.socket,'connect',side_effect=AssertionError('network forbidden'))
        p.start();self.addCleanup(p.stop)
        self.path=install(self.root,document(calls=8))
        self.budget=PersistentBudget(self.root,scope=SCOPE,billing_scope=BILLING,policy=BudgetPolicy(max_attempts=8))

    def test_actual_semantic_graph_uses_same_seven_slots_without_double_charging(self):
        from test_orchestration_semantic import PhaseBSemanticGraphTests
        from test_context import AS_OF
        fixture=PhaseBSemanticGraphTests();fixture.setUp()
        service,_,_,provider=fixture.make(judgments=('uncertain','relevant'),semantics=('incomplete','supported'))
        with self.budget.reserve(WorkflowRequest('graph',GRAPH,'Who?','draft_review','custom',AS_OF)) as r:
            result=service.ask('Who?',mode='draft_review',reference_date=AS_OF,reservation=r)
        self.assertTrue(result.accepted)
        self.assertEqual(len(provider.calls),7)
        self.assertEqual(self.budget.check_ready().attempts_consumed,7)
        self.assertEqual(service.last_trace.retrieval_attempts,2)
        self.assertEqual(result.public.coverage.status,'not_assessed')

    def test_qa_validates_binding_and_publishes_only_accepted_answer(self):
        from test_qa import FakeLLM, FixtureStore, authored_claims
        from test_context import AS_OF, chunk, hit
        from northstar_atlas.generation import prepare_answer
        from northstar_atlas.qa import QAService
        hits=(hit(chunk()),)
        provider=FakeLLM(authored_claims(prepare_answer('Who?',hits,mode='draft_review',reference_date=AS_OF)))
        service=QAService(FixtureStore(hits),None,provider,corpus_fingerprint='fixture')
        with self.budget.reserve(WorkflowRequest('qa',LINEAR,'Who?','draft_review','custom',AS_OF)) as r:
            mismatch=service.ask('Another?',mode='draft_review',reference_date=AS_OF,reservation=r)
            self.assertFalse(mismatch.accepted);self.assertFalse(provider.calls)
            result=service.ask('Who?',mode='draft_review',reference_date=AS_OF,reservation=r)
        self.assertTrue(result.accepted);self.assertEqual(len(provider.calls),1)

    def test_nebius_requires_active_permit_before_mock_transport(self):
        from northstar_atlas.nebius import NebiusProvider, NebiusSettings, ProviderError
        sends=[]
        def send(req):
            sends.append(req)
            self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['calls'][0]['status'],'started')
            return httpx.Response(200,json={'model':MODEL_ID,'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'{}'}}],
                'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}},request=req)
        with NebiusProvider(NebiusSettings('synthetic-fixture-key'),allow_paid=True,requires_budget=True,
                            transport=httpx.MockTransport(send)) as provider:
            with self.assertRaises(ProviderError):provider.generate(request())
            with self.budget.reserve(WorkflowRequest('http',LINEAR,'Who?','approved_only','chroma',date.today())) as r:
                r.invoke(provider,'generator',0,request())
            with self.assertRaises(ProviderError):provider.generate_budgeted(request(),r)
        self.assertEqual(len(sends),1)

    def test_ui_reserves_before_setup_duplicate_across_owners_is_rejected(self):
        from northstar_atlas.ui.runtime import ResourceOwner, Submission
        from test_ui import success
        calls=[]
        class Service:
            def ask(inner,question,**kwargs):
                kwargs['reservation'].invoke(Provider(),'generator',0,request())
                calls.append(question)
                return success()
        @contextmanager
        def opener(root,**kwargs):
            self.assertEqual(json.loads(self.path.read_text())['workflows'][-1]['status'],'reserved')
            yield Service()
        submission=Submission('turn-one','Who?','approved_only','chroma',date.today())
        for expected in ('answer','error'):
            owner=ResourceOwner(self.root,live_enabled=True,opener=opener,budget=self.budget)
            self.addCleanup(owner.close)
            self.assertEqual(owner.submit(submission,consent=True).kind,expected)
        self.assertEqual(calls,['Who?'])

    def test_budget_readiness_precedes_credential_and_retriever_access(self):
        from northstar_atlas.qa import open_qa_service, QAConfigurationError
        self.path.write_text('{}')
        with patch('northstar_atlas.nebius.load_settings',side_effect=AssertionError('credentials accessed')), \
             patch('northstar_atlas.qa.open_retriever',side_effect=AssertionError('retrieval accessed')):
            with self.assertRaises(QAConfigurationError):
                with open_qa_service(self.root,allow_paid=True,budget=self.budget):pass

    def test_graph_live_gate_remains_closed_even_with_healthy_budget(self):
        from northstar_atlas.ui.runtime import ResourceOwner, Submission, LANGGRAPH_WORKFLOW
        owner=ResourceOwner(self.root,live_enabled=True,budget=self.budget)
        self.addCleanup(owner.close)
        result=owner.submit(Submission('g','Who?','draft_review','chroma',date.today(),workflow=LANGGRAPH_WORKFLOW),consent=True)
        self.assertEqual(result.code,'workflow_disabled')
        self.assertFalse(json.loads(self.path.read_text())['workflows'])

    def test_transport_permit_cannot_be_reused_in_one_stage(self):
        from northstar_atlas.nebius import NebiusProvider, NebiusSettings, ProviderError
        sends=[]
        def send(req):
            sends.append(req)
            return httpx.Response(200,json={'model':MODEL_ID,'choices':[{'finish_reason':'stop',
                'message':{'role':'assistant','content':'{}'}}],
                'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}},request=req)
        with NebiusProvider(NebiusSettings('synthetic-fixture-key'),allow_paid=True,requires_budget=True,
                            transport=httpx.MockTransport(send)) as real:
            class Twice:
                def generate_budgeted(inner, req, permit):
                    response=real.generate_budgeted(req,permit)
                    with self.assertRaises(ProviderError):real.generate_budgeted(req,permit)
                    return response
            with self.budget.reserve(WorkflowRequest('once',LINEAR,'Who?','approved_only','chroma',date.today())) as r:
                r.invoke(Twice(),'generator',0,request())
        self.assertEqual(len(sends),1)

    def test_ui_withholds_success_when_settlement_cannot_be_saved(self):
        from northstar_atlas.ui.runtime import ResourceOwner, Submission
        from test_ui import success
        class Service:
            def ask(inner,question,**kwargs):
                kwargs['reservation'].invoke(Provider(),'generator',0,request())
                return success()
        @contextmanager
        def opener(root,**kwargs):yield Service()
        owner=ResourceOwner(self.root,live_enabled=True,opener=opener,budget=self.budget)
        self.addCleanup(owner.close)
        write=self.budget._write
        def fail_settlement(value):
            if value['workflows'][-1]['status']=='closed':raise OSError('synthetic private detail')
            write(value)
        with patch.object(self.budget,'_write',side_effect=fail_settlement):
            result=owner.submit(Submission('failed-write','Who?','approved_only','chroma',date.today()),consent=True)
        self.assertEqual(result.kind,'error')
        self.assertEqual(result.code,'budget_unavailable')
        self.assertNotIn('private detail',result.text)
        self.assertFalse(result.citations)
        self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['status'],'reserved')
