"""Real graph transitions with authored evidence/LLM responses, no quality claims."""
from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, chunk, family, hit
from test_bm25 import catalog
from test_qa import FakeLLM, FixtureStore, authored_claims
from test_orchestration_assessment import route
from northstar_atlas import generation as g, orchestration as o, qa
from northstar_atlas.context import build_context
from northstar_atlas.orchestration_assessment import PROFILE, PhaseALimits
from northstar_atlas.grounding import EvidenceObligation
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.llm import GenerationLimits, GenerationResponse, QwenTokenBound


class ScopedStore(FixtureStore):
    def search(self, question, embedder, *, department_scope=None, **kwargs):
        return super().search(question, embedder, department_scope=department_scope, **kwargs)


class PhaseAGraphTests(unittest.TestCase):
    def setUp(self):
        self.source = chunk()
        self.hits = (hit(self.source),)
        self.cat = catalog([self.source])

    def make(self, *, first=None, second=None, routing=None, judgments=(), body=None,
             cat=None, plan=('dense','bm25'), phase_limits=PhaseALimits(), provider=None):
        first = ScopedStore(self.hits) if first is None else first
        second = ScopedStore(self.hits) if second is None else second
        grade_count = 0
        def reply(request):
            nonlocal grade_count
            data = json.loads(request.user)
            if 'chunks' in data:
                choice = judgments[grade_count] if grade_count < len(judgments) else 'relevant'
                grade_count += 1
                if callable(choice): return choice(data)
                return {'schema_version':'relevance-v1', 'judgments':[
                    {'chunk_id':row['chunk_id'], 'relevance':choice} for row in data['chunks']]}
            if set(data) == {'question'}:
                return route() if routing is None else routing
            if body is not None: return body
            prepared = g.prepare_answer('Who?', self.hits, mode='draft_review', reference_date=AS_OF)
            return authored_claims(prepared)
        provider = provider or FakeLLM(reply)
        service = o.GraphQAService({'dense':first,'bm25':second,'hybrid':first}, None, provider,
            catalog=cat or self.cat, strategy_plan=plan, profile=PROFILE, phase_limits=phase_limits)
        return service, first, second, provider

    def ask(self, service, **kwargs):
        return service.ask('Who?',mode='draft_review',reference_date=AS_OF,**kwargs)

    def failure(self, result, code):
        self.assertIsInstance(result,qa.QAFailure)
        self.assertEqual(result.public.code,code)
        self.assertNotIn('SECRET_REJECTED',json.dumps(asdict(result),default=str))
        self.assertFalse(hasattr(result.public,'text'))

    def test_normal_three_calls_same_public_contract_and_aggregate_usage(self):
        service, first, second, provider = self.make()
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.status,'not_assessed')
        self.assertEqual(result.diagnostics.provider_calls,3)
        self.assertEqual(result.diagnostics.usage.total_tokens,450)
        self.assertEqual([c.purpose for c in service.last_trace.calls],['router','grader','generator'])
        self.assertEqual(service.last_trace.nodes,('router','retriever','assessor','generator','validator'))
        self.assertEqual(second.calls,[])
        self.assertIn('(proposed)',result.public.text)
        self.assertEqual(first.calls[0][2]['department_scope'],None)
        # Invocation counters reset; service resources are reusable.
        self.assertTrue(self.ask(service).accepted)
        self.assertEqual(len(provider.calls),6)

    def test_alternate_broadens_scope_changes_strategy_and_caps_four_calls(self):
        platform = replace(chunk(key='platform', doc='PLAT-001'),
                           department='Platform Engineering / DevOps')
        sources = (self.source, platform)
        service, first, second, provider = self.make(routing=route(['platform_engineering']),
                                                     judgments=('uncertain','relevant'),
                                                     first=ScopedStore((hit(platform),)),
                                                     second=ScopedStore(self.hits),
                                                     cat=catalog(sources))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(len(provider.calls),4)
        self.assertEqual(first.calls[0][2]['department_scope'],('Platform Engineering / DevOps',))
        self.assertIsNone(second.calls[0][2]['department_scope'])
        self.assertEqual([a.strategy for a in service.last_trace.attempts],['dense','bm25'])
        self.assertEqual([c.attempt for c in service.last_trace.calls if c.purpose=='grader'],[1,2])
        self.assertEqual(service.last_trace.retrieval_attempts,2)

    def test_all_negative_or_uncertain_exhaustion_abstains_without_generation(self):
        for value in ('not_relevant','uncertain'):
            service, _, _, provider = self.make(judgments=(value,value))
            result = self.ask(service)
            self.assertTrue(result.accepted)
            self.assertEqual(result.public.kind,'abstain')
            self.assertEqual(result.public.presentation.abstention_basis,'insufficient_evidence')
            self.assertEqual(len(provider.calls),3)
            self.assertNotIn('generator',[c.purpose for c in service.last_trace.calls])
            self.assertEqual(result.public.citations,())

    def test_relevance_grading_cannot_waive_missing_reviewed_evidence(self):
        package = build_context(self.hits, counter=QwenTokenBound(), max_tokens=28000,
                                mode='draft_review', reference_date=AS_OF)
        obligation = EvidenceObligation('required_source', package.groups[0].citation,
                                        self.source.text.strip(), validation_mode='extractive')
        other = chunk('Other evidence.\n', key='other', doc='SEC-001', section='9.1')
        service, _, _, provider = self.make(
            first=ScopedStore(self.hits), second=ScopedStore((hit(other),)),
            cat=catalog((self.source, other)), judgments=('not_relevant',))
        result = self.ask(service, obligations=(obligation,))
        self.failure(result, 'context_failed')
        self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertEqual([call.purpose for call in service.last_trace.calls], ['router', 'grader'])
        self.assertEqual(len(provider.calls), 2)

    def test_empty_retrieval_skips_grading_but_keeps_router_accounting(self):
        service, _, _, provider = self.make(first=ScopedStore(),second=ScopedStore())
        self.assertEqual(self.ask(service).public.kind,'abstain')
        self.assertEqual(len(provider.calls),1)

    def test_clarification_remains_zero_calls(self):
        service, first, second, provider = self.make()
        hint=o.CallerReviewedClarification('Who?','draft_review','Which environment?',('Access environment.',))
        result=self.ask(service,clarification=hint)
        self.assertEqual(result.public.kind,'clarify')
        self.assertEqual(first.calls+second.calls+provider.calls,[])

    def test_relevant_sibling_retains_whole_group_and_original_direct_rank(self):
        original=chunk('Rule.\nRequired exception.\n'); pieces=family(original,[6])
        self.hits=(hit(pieces[0],pieces[1:]),)
        def mixed(data):
            return {'schema_version':'relevance-v1','judgments':[
                {'chunk_id':row['chunk_id'],'relevance':'not_relevant' if row['chunk_id']==pieces[0].chunk_id else 'relevant'}
                for row in data['chunks']]}
        service, _, _, _=self.make(cat=catalog([original],pieces),judgments=(mixed,))
        result=self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(service.last_trace.gradings[0].retained_context_ids,(pieces[0].chunk_id,))
        self.assertEqual(service.last_trace.attempts[0].direct_ids,(pieces[0].chunk_id,))
        self.assertIn('Required exception.',result.public.text)

    def test_unresolved_group_prevents_generating_from_other_relevant_group(self):
        second=chunk('Other condition.\n',key='two',doc='IT-001',section='4.2')
        hits=(*self.hits,hit(second))
        def mixed(data):
            return {'schema_version':'relevance-v1','judgments':[
                {'chunk_id':row['chunk_id'],'relevance':'relevant' if row['chunk_id']==self.source.chunk_id else 'uncertain'}
                for row in data['chunks']]}
        service, _, _, provider=self.make(cat=catalog([self.source,second]),first=ScopedStore(hits),
                                         judgments=(mixed,),plan=('dense',))
        self.assertEqual(self.ask(service).public.kind,'abstain')
        self.assertEqual(len(provider.calls),2)

    def test_missing_sibling_and_context_budget_fail_before_grading(self):
        original=chunk('First.\nSecond.\n'); pieces=family(original,[7])
        service, _, second, provider=self.make(cat=catalog([original],pieces),first=ScopedStore([hit(pieces[0])]))
        self.failure(self.ask(service),'context_failed')
        self.assertEqual(len(provider.calls),1); self.assertEqual(second.calls,[])
        service, _, second, provider=self.make(phase_limits=PhaseALimits(grading_context_tokens=5000))
        self.failure(self.ask(service),'context_failed')
        self.assertEqual(len(provider.calls),1); self.assertEqual(second.calls,[])

    def test_bad_route_or_grading_withheld_without_alternate(self):
        for routing, judgments in (('SECRET_REJECTED',()),(route(),(lambda _: 'SECRET_REJECTED',)),
                                   (route(),(lambda _: {'schema_version':'relevance-v1','judgments':[]},))):
            service, _, second, provider=self.make(routing=routing,judgments=judgments)
            self.failure(self.ask(service),'response_rejected')
            self.assertLessEqual(len(provider.calls),2); self.assertEqual(second.calls,[])

    def test_provider_error_and_bad_envelope_distinguished(self):
        for provider, expected in ((FakeLLM(error=TimeoutError('SECRET_REJECTED')),'provider_failed'),
                                  (FakeLLM(), 'response_rejected')):
            if expected=='response_rejected':
                provider.generate=lambda r: GenerationResponse('{}',r.model,'length',None,.1,1)
            service, first, second, _=self.make(provider=provider)
            result=self.ask(service); self.failure(result,expected)
            self.assertEqual(result.diagnostics.provider_calls,1)
            self.assertEqual(first.calls+second.calls,[])

    def test_invalid_generation_is_never_published_or_retried(self):
        service, _, second, provider=self.make(body='SECRET_REJECTED')
        self.failure(self.ask(service),'response_rejected')
        self.assertEqual(len(provider.calls),3); self.assertEqual(second.calls,[])

    def test_draft_or_out_of_scope_direct_evidence_rejected(self):
        service, _, _, provider=self.make()
        self.failure(service.ask('Who?',mode='approved_only',reference_date=AS_OF),'context_failed')
        self.assertEqual(len(provider.calls),1)
        service, _, _, provider=self.make(routing=route(['human_resources']))
        self.failure(self.ask(service),'context_failed')
        self.assertEqual(len(provider.calls),1)

    def test_limits_and_unsupported_adapter_rejected_before_calls(self):
        service, first, second, provider=self.make()
        for limits in (GenerationLimits(context_tokens=65536),GenerationLimits(output_tokens=4096),
                       GenerationLimits(margin_tokens=1)):
            self.failure(self.ask(service,limits=limits),'configuration_failed')
        self.assertEqual(first.calls+second.calls+provider.calls,[])
        with self.assertRaises(qa.QAConfigurationError): self.make(first=FixtureStore(self.hits))

    def test_factory_disallows_paid_or_remote_activation_before_io(self):
        with patch.object(o,'load_catalog',side_effect=AssertionError('catalog accessed')) as loader, \
             patch.object(o,'open_retriever',side_effect=AssertionError('backend opened')) as opener:
            for kwargs in ({},{'allow_paid':True},{'provider':FakeLLM(),'allow_paid':True},
                           {'provider':FakeLLM(),'backend':'pinecone'}):
                with self.subTest(kwargs=kwargs),self.assertRaises(qa.QAConfigurationError):
                    with o.open_graph_qa_service(ROOT,snapshot_id='unused',profile=PROFILE,**kwargs): pass
            loader.assert_not_called(); opener.assert_not_called()

    def test_unknown_usage_does_not_report_partial_sum(self):
        service, _, _, provider=self.make()
        generate=provider.generate
        provider.generate=lambda r: replace(generate(r),usage=None)
        result=self.ask(service)
        self.assertTrue(result.accepted)
        self.assertIsNone(result.diagnostics.usage)
        self.assertTrue(service.last_trace.usage_unknown)
