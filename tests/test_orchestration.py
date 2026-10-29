"""Bounded graph contracts with authored fixtures, never retrieval-quality claims."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import importlib
import importlib.util
import json
import socket
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_context import AS_OF, MODEL, ROOT, chunk, family, hit
from test_bm25 import catalog
from test_qa import FakeLLM, FixtureStore, authored_claims
from test_generation_v2 import payload
from test_abstention import authored_retention, retention_requirements
from test_extractive_profile import QUESTION, refs_payload, terraform_scope
from test_grounding import terraform_obligations
from northstar_atlas import generation as g, qa
from northstar_atlas.grounding import EvidenceObligation
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, load_documents
from northstar_atlas.llm import GenerationLimits, GenerationResponse, TokenUsage


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('northstar_atlas.orchestration'),
                             'optional five-node service is not implemented')
        self.o = importlib.import_module('northstar_atlas.orchestration')
        guard = patch.object(socket.socket, 'connect', side_effect=AssertionError('network forbidden'))
        guard.start(); self.addCleanup(guard.stop)
        self.source = chunk()
        self.cat = catalog([self.source])

    def make(self, *, first=None, second=None, cat=None, body=None, plan=('dense', 'bm25'), provider=None):
        cat = cat or self.cat
        first = FixtureStore([hit(self.source)]) if first is None else first
        second = FixtureStore() if second is None else second
        if provider is None:
            p = g.prepare_answer('Who?', [hit(self.source)], mode='draft_review', reference_date=AS_OF)
            provider = FakeLLM(authored_claims(p) if body is None else body)
        stores = {'dense': first, 'bm25': second, 'hybrid': first}
        service = self.o.GraphQAService(stores, None, provider, catalog=cat, strategy_plan=plan)
        return service, first, second, provider

    def ask(self, service, **kwargs):
        return service.ask('Who?', mode='draft_review', reference_date=AS_OF, **kwargs)

    def error(self, result, code):
        self.assertIsInstance(result, qa.QAFailure)
        self.assertEqual(result.public.code, code)
        self.assertNotIn('SECRET_REJECTED', json.dumps(asdict(result), default=str))
        self.assertFalse(hasattr(result.public, 'text'))

    def test_five_node_success_reuses_linear_public_contract(self):
        service, first, second, provider = self.make()
        result = self.ask(service)
        linear = qa.QAService(first, None, provider, corpus_fingerprint='fixture')
        self.assertEqual(result.public, self.ask(linear).public)
        self.assertEqual(service.last_trace.nodes, ('router', 'retriever', 'assessor', 'generator', 'validator'))
        self.assertEqual(service.last_trace.retrieval_attempts, 1)
        self.assertEqual(result.diagnostics.provider_calls, 1)
        self.assertEqual(result.public.coverage.status, 'not_assessed')
        self.assertEqual(service.last_trace.assessments[-1].semantic_coverage, 'not_assessed')
        self.assertEqual(second.calls, [])

    def test_one_alternate_replaces_empty_evidence_and_preserves_options(self):
        service, first, second, provider = self.make(first=FixtureStore(), second=FixtureStore([hit(self.source)]))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertEqual(service.last_trace.nodes, ('router', 'retriever', 'assessor', 'retriever', 'assessor', 'generator', 'validator'))
        self.assertEqual(first.calls[0], second.calls[0])
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(result.diagnostics.retrieval_strategy, 'bm25')

    def test_two_empty_attempts_abstain_without_inference(self):
        service, first, second, provider = self.make(first=FixtureStore(), second=FixtureStore())
        result = self.ask(service)
        self.assertEqual((len(first.calls), len(second.calls), len(provider.calls)), (1, 1, 0))
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(result.public.presentation.abstention_basis, 'insufficient_evidence')
        self.assertEqual(result.public.citations, ())
        self.assertIn('retrieval may be incomplete', result.public.text)

    def test_explicit_strategy_plans_and_single_attempt(self):
        for plan in (('dense',), ('bm25',), ('hybrid',), ('hybrid', 'dense'), ('bm25', 'dense')):
            with self.subTest(plan=plan):
                service, first, second, provider = self.make(first=FixtureStore(), plan=plan)
                result = self.ask(service)
                self.assertEqual(result.public.kind, 'abstain')
                self.assertEqual(service.last_trace.retrieval_attempts, len(plan))
                self.assertEqual(len(provider.calls), 0)

    def test_invalid_plan_cannot_add_third_attempt_or_silent_fallback(self):
        for plan in ((), ('dense', 'dense'), ('dense', 'bm25', 'hybrid'), ('unknown',), ['dense']):
            with self.subTest(plan=plan), self.assertRaises(qa.QAConfigurationError): self.make(plan=plan)
        with self.assertRaises(qa.QAConfigurationError):
            self.o.GraphQAService({'dense': FixtureStore()}, None, FakeLLM(), catalog=self.cat)

    def test_invalid_input_precedes_retrieval_and_inference(self):
        service, first, second, provider = self.make()
        for question, options in (('', {}), ('x', {'mode':'all'}), ('x', {'top_k':6}),
                                  ('x', {'top_k':True}), ('x', {'reference_date':'today'}),
                                  ('x', {'limits':None})):
            self.error(service.ask(question, **options), 'invalid_input')
        self.assertEqual(first.calls + second.calls + provider.calls, [])

    def test_scoped_caller_clarification_has_zero_io_and_no_policy_claims(self):
        service, first, second, provider = self.make()
        hint = self.o.CallerReviewedClarification('Who?', 'draft_review', 'Which access environment?', ('Access environment.',))
        result = self.ask(service, clarification=hint)
        self.assertEqual(result.public.kind, 'clarify')
        self.assertEqual(result.public.citations, ())
        self.assertEqual(result.public.presentation.clarification_question, 'Which access environment?')
        self.assertEqual(first.calls + second.calls + provider.calls, [])
        self.assertEqual(service.last_trace.nodes, ('router', 'validator'))
        self.error(self.ask(service, clarification=replace(hint, question='Different')), 'configuration_failed')

    def test_generated_clarification_still_uses_grounding_contract(self):
        service, _, _, _ = self.make(body=payload(kind='clarify'))
        result = self.ask(service)
        self.assertEqual(result.public.kind, 'clarify')
        self.assertEqual(result.diagnostics.provider_calls, 1)

    def test_retrieval_and_provider_failures_are_errors_without_alternate_or_retry(self):
        service, first, second, provider = self.make(first=FixtureStore(error=RuntimeError('SECRET_REJECTED')))
        self.error(self.ask(service), 'retrieval_failed')
        self.assertEqual(second.calls + provider.calls, [])
        service, first, second, provider = self.make(provider=FakeLLM(error=TimeoutError('SECRET_REJECTED')))
        result = self.ask(service)
        self.error(result, 'provider_failed')
        self.assertEqual(result.diagnostics.provider_calls, 1)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(second.calls, [])

    def test_malformed_truncated_and_invalid_citation_candidates_withheld(self):
        prepared = g.prepare_answer('Who?', [hit(self.source)], mode='draft_review', reference_date=AS_OF)
        invalid = authored_claims(prepared)
        invalid['main_answer']['claims'][0]['support'][0]['citation'] = 'SECRET_REJECTED'
        for body in ('SECRET_REJECTED', invalid):
            service, _, second, provider = self.make(body=body)
            result = self.ask(service)
            self.error(result, 'response_rejected')
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual(second.calls, [])
        provider = FakeLLM()
        provider.generate = lambda r: GenerationResponse('{}', r.model, 'length', None, .1, 1)
        service, _, second, _ = self.make(provider=provider)
        self.error(self.ask(service), 'response_rejected')
        self.assertEqual(second.calls, [])

    def test_missing_sibling_and_stale_payload_stop_before_alternate(self):
        original = chunk('First condition.\nSecond condition.\n')
        pieces = family(original, [17])
        c = catalog([original], pieces)
        for bad in (hit(pieces[0]), hit(replace(pieces[0], content_hash='b'*64), pieces[1:]),
                    replace(hit(pieces[0], pieces[1:]), model=replace(MODEL, revision='stale'))):
            service, _, second, provider = self.make(cat=c, first=FixtureStore([bad]))
            self.error(self.ask(service), 'context_failed')
            self.assertEqual(second.calls + provider.calls, [])

    def test_approved_only_rejects_adapter_draft_leak_without_changing_mode(self):
        service, first, second, provider = self.make()
        self.error(service.ask('Who?', reference_date=AS_OF), 'context_failed')
        self.assertEqual(first.calls[0][2]['mode'], 'approved_only')
        self.assertEqual(second.calls + provider.calls, [])

    def test_prompt_overflow_does_not_drop_evidence_or_try_alternate(self):
        service, _, second, provider = self.make()
        self.error(self.ask(service, limits=GenerationLimits(context_tokens=4096, output_tokens=100, margin_tokens=10)), 'context_failed')
        self.assertEqual(second.calls + provider.calls, [])

    def obligation(self):
        p = g.prepare_answer('Who?', [hit(self.source)], mode='draft_review', reference_date=AS_OF)
        return EvidenceObligation('scope', p.context.groups[0].citation, self.source.text.strip(), 'extractive')

    def test_missing_reviewed_source_retries_without_injecting_preflight_evidence(self):
        ob = self.obligation()
        unrelated = chunk('An unrelated policy.', key='other', doc='SEC-001')
        service, first, second, provider = self.make(cat=catalog([self.source, unrelated]),
            first=FixtureStore([hit(unrelated)]), second=FixtureStore([hit(self.source)]), body=payload((ob,)))
        result = self.ask(service, obligations=(ob,))
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.satisfied_ids, ('scope',))
        self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertNotIn('An unrelated policy.', provider.calls[0].user)
        service, _, _, provider = self.make(first=FixtureStore(), body=payload((ob,)))
        self.error(self.ask(service, obligations=(ob,)), 'context_failed')
        self.assertEqual(provider.calls, [])

    def test_invalid_reviewed_input_fails_preflight_without_search(self):
        ob = self.obligation()
        for bad in (replace(ob, source_quote='Invented rule.'),
                    replace(ob, citation=replace(ob.citation, content_hash='b'*64))):
            service, first, second, provider = self.make()
            self.error(self.ask(service, obligations=(bad,)), 'configuration_failed')
            self.assertEqual(first.calls + second.calls + provider.calls, [])

    def test_documented_gap_preserves_reviewed_evidence_and_public_status(self):
        source = next(c for d in load_documents(ROOT, mode='draft_review') if d.document_id=='COMP-001'
                      for c in chunk_document(d).chunks if c.section_id=='11.1')
        p = g.prepare_answer('Who?', [hit(source)], mode='draft_review', reference_date=AS_OF)
        requirements = retention_requirements(p.context)
        service, _, _, _ = self.make(cat=catalog([source]), first=FixtureStore([hit(source)]), body=authored_retention(requirements))
        result = self.ask(service, abstention=requirements)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(result.public.presentation.abstention_basis, 'documented_gap')
        self.assertEqual(result.public.coverage.evidence_linked_ids, ('no_period_established',))
        self.assertIn('COMP-001 v0.1 §11.1 (proposed)', result.public.text)

    def test_terraform_application_owned_requirements_and_no_automatic_selection(self):
        source = next(c for d in load_documents(ROOT, mode='draft_review') if d.document_id=='PLAT-001'
                      for c in chunk_document(d).chunks if c.section_id=='6.1')
        p = g.prepare_answer(QUESTION, [hit(source)], mode='draft_review', reference_date=AS_OF)
        obs = terraform_obligations(p.context)
        service, _, _, _ = self.make(cat=catalog([source]), first=FixtureStore([hit(source)]), body=refs_payload(obs))
        result = service.ask(QUESTION, mode='draft_review', reference_date=AS_OF, obligations=obs, extractive_scope=terraform_scope())
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.extractive_ids, tuple(o.obligation_id for o in obs))
        self.assertEqual(len(result.diagnostics.application_owned_ids), 4)
        ordinary, _, _, _ = self.make(cat=catalog([source]), first=FixtureStore([hit(source)]), body=authored_claims(p))
        result = ordinary.ask(QUESTION, mode='draft_review', reference_date=AS_OF)
        self.assertEqual(result.public.coverage.status, 'not_assessed')

    def test_fresh_invocation_counters_mode_and_evidence(self):
        hr = chunk(key='hr', doc='HR-001', status=APPROVED, effective=AS_OF)
        first = FixtureStore([hit(hr)])
        p = g.prepare_answer('Who?', first.hits, reference_date=AS_OF)
        service, _, _, provider = self.make(cat=catalog([hr]), first=first, body=authored_claims(p))
        self.assertTrue(service.ask('Who?', reference_date=AS_OF).accepted)
        first.hits = ()
        result = self.ask(service)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(result.diagnostics.provider_calls, 0)
        self.assertEqual(result.diagnostics.evidence, ())
        self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertEqual(len(provider.calls), 1)

    def test_tracing_environment_cannot_create_client_or_export_state(self):
        import langsmith
        service, _, _, _ = self.make()
        with patch.dict('os.environ', {'LANGSMITH_TRACING':'true', 'LANGCHAIN_TRACING_V2':'true'}), \
             patch.object(langsmith.Client, '__init__', side_effect=AssertionError('tracing forbidden')):
            result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertNotIn('Only within', json.dumps(asdict(service.last_trace), default=str))

    def test_inherited_callbacks_cannot_receive_prompts_or_state(self):
        from langchain_core.callbacks import BaseCallbackHandler
        from langchain_core.runnables.config import var_child_runnable_config
        events = []
        class Spy(BaseCallbackHandler):
            def on_chain_start(self, *args, **kwargs): events.append(args)
        token = var_child_runnable_config.set({'callbacks': [Spy()]})
        try:
            service, _, _, _ = self.make()
            self.assertTrue(self.ask(service).accepted)
        finally:
            var_child_runnable_config.reset(token)
        self.assertEqual(events, [])

    def test_recursion_and_repeated_inference_guards_withhold_output(self):
        from langgraph.errors import GraphRecursionError
        service, first, _, provider = self.make()
        with patch.object(service._graph, 'invoke', side_effect=GraphRecursionError('SECRET_REJECTED')):
            self.error(self.ask(service), 'internal_error')
        self.assertEqual(first.calls + provider.calls, [])
        # Fault injection at the orchestration boundary: a second generation
        # invocation must not result in a second underlying provider call.
        actual = g.generate_answer
        def twice(prepared, observed):
            actual(prepared, observed)
            return actual(prepared, observed)
        with patch.object(self.o, 'generate_answer', side_effect=twice):
            result = self.ask(service)
        self.error(result, 'response_rejected')
        self.assertEqual(len(provider.calls), 1)

    def test_legacy_extractive_and_linked_obligations_remain_distinct(self):
        ob = self.obligation()
        for mode, accepted in (('extractive', False), ('evidence_linked', True)):
            checked = replace(ob, validation_mode=mode)
            body = payload((checked,))
            body['main_answer']['obligations']['scope']['statement'] = 'The action remains bounded by its authorization.'
            service, _, second, _ = self.make(body=body)
            result = self.ask(service, obligations=(checked,))
            self.assertEqual(result.accepted, accepted)
            if accepted:
                self.assertEqual(result.public.coverage.evidence_linked_ids, ('scope',))
            else:
                self.assertEqual(result.public.code, 'response_rejected')
            self.assertEqual(second.calls, [])

    def test_reviewed_scope_cannot_be_bypassed_by_clarification_or_wrong_mode(self):
        ob = self.obligation()
        service, first, second, provider = self.make()
        hint = self.o.CallerReviewedClarification('Who?', 'draft_review', 'Which scope?', ('Scope.',))
        self.error(self.ask(service, obligations=(ob,), clarification=hint), 'configuration_failed')
        self.error(service.ask('Who?', reference_date=AS_OF, obligations=(ob,)), 'configuration_failed')
        self.assertEqual(first.calls + second.calls + provider.calls, [])

    def test_missing_ids_quotes_and_authority_labels_remain_rejected(self):
        ob = self.obligation()
        for change in ('missing', 'extra', 'quote', 'status'):
            body = payload((ob,))
            if change == 'missing': body['main_answer']['obligations'] = {}
            elif change == 'extra': body['main_answer']['obligations']['invented'] = {'action':'include_reviewed'}
            elif change == 'quote': body['main_answer']['obligations']['scope']['support'][0]['quote'] = 'SECRET_REJECTED invented rule.'
            else:
                support = body['main_answer']['obligations']['scope']['support'][0]
                support['citation'] = support['citation'].replace('(proposed)', '(approved, synthetic)')
            with self.subTest(change=change):
                service, _, second, provider = self.make(body=body)
                self.error(self.ask(service, obligations=(ob,)), 'response_rejected')
                self.assertEqual(len(provider.calls), 1)
                self.assertEqual(second.calls, [])

    def test_complete_siblings_are_delivered_without_context_trimming(self):
        original = chunk('First condition.\nSecond condition.\n')
        pieces = family(original, [17])
        hits = (hit(pieces[0], pieces[1:]),)
        p = g.prepare_answer('Who?', hits, mode='draft_review', reference_date=AS_OF)
        service, _, _, provider = self.make(cat=catalog([original], pieces), first=FixtureStore(hits), body=authored_claims(p))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.diagnostics.evidence), 2)
        self.assertIn('Second condition.', provider.calls[0].user)
        self.assertEqual(result.diagnostics.evidence[0].required_context_ids, (pieces[1].chunk_id,))

    def test_factory_shares_dense_connection_with_hybrid_and_closes_once(self):
        opened, closed = [], []
        dense = FixtureStore([hit(self.source)])
        @contextmanager
        def retriever(root, backend):
            opened.append(backend)
            try: yield dense, SimpleNamespace(model=MODEL), 'fixture'
            finally: closed.append(backend)
        p = g.prepare_answer('Who?', dense.hits, mode='draft_review', reference_date=AS_OF)
        with patch.object(self.o, 'load_catalog', return_value=self.cat), \
             patch.object(self.o, 'open_retriever', retriever), \
             patch.dict('os.environ', {'VECTOR_STORE':'chroma'}):
            with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, provider=FakeLLM(authored_claims(p)),
                                             strategy_plan=('hybrid', 'dense')) as service:
                for _ in range(2): self.assertTrue(self.ask(service).accepted)
                self.assertEqual(service._stores['hybrid'].dense, service._stores['dense'])
        self.assertEqual(opened, ['chroma'])
        self.assertEqual(closed, ['chroma'])
        self.assertEqual([c[2]['top_k'] for c in dense.calls], [20, 20])

    def test_factory_bm25_only_uses_no_dense_model_or_connection(self):
        p = g.prepare_answer('scope?', [hit(self.source)], mode='draft_review', reference_date=AS_OF)
        with patch.object(self.o, 'load_catalog', return_value=self.cat), \
             patch.object(self.o, 'open_retriever', side_effect=AssertionError('no backend initialization')):
            with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, strategy_plan=('bm25',),
                                             provider=FakeLLM(authored_claims(p))) as service:
                result = service.ask('scope?', mode='draft_review', reference_date=AS_OF)
        self.assertTrue(result.accepted)
        self.assertEqual(result.diagnostics.retrieval_strategy, 'bm25')

    def test_factory_invalid_config_and_missing_paid_opt_in_precede_io(self):
        from northstar_atlas import nebius
        with patch.object(self.o, 'load_catalog', side_effect=AssertionError('catalog IO')), \
             patch.object(self.o, 'open_retriever', side_effect=AssertionError('backend IO')), \
             patch.object(nebius, 'load_settings', side_effect=AssertionError('credential IO')):
            for options in ({}, {'provider':object()}, {'provider':FakeLLM(), 'backend':'other'},
                            {'provider':FakeLLM(), 'strategy_plan':('dense','dense')}):
                with self.subTest(options=options), self.assertRaises(qa.QAConfigurationError):
                    with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, **options):
                        self.fail('invalid configuration opened')

    def test_factory_missing_credentials_is_safe_without_backend_io(self):
        from northstar_atlas import nebius
        with patch.object(self.o, 'load_catalog', return_value=self.cat), \
             patch.object(self.o, 'open_retriever') as opened, \
             patch.object(nebius, 'load_settings', side_effect=RuntimeError('SECRET_REJECTED')):
            with self.assertRaises(qa.QAConfigurationError) as caught:
                with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, allow_paid=True): self.fail('opened')
        self.assertNotIn('SECRET_REJECTED', str(caught.exception))
        opened.assert_not_called()

    def test_factory_pinecone_is_explicit_and_fingerprint_mismatch_closes(self):
        closed, opened = [], []
        @contextmanager
        def retriever(root, backend, *, snapshot_id):
            self.assertEqual(snapshot_id, 'a'*64)
            opened.append(backend)
            try: yield FixtureStore(), SimpleNamespace(model=MODEL), 'wrong-fingerprint'
            finally: closed.append(backend)
        with patch.object(self.o, 'load_catalog', return_value=self.cat), patch.object(self.o, 'open_retriever', retriever):
            with self.assertRaises(qa.QAConfigurationError):
                with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, backend='pinecone', provider=FakeLLM()): self.fail('opened')
        self.assertEqual(opened, ['pinecone'])
        self.assertEqual(closed, ['pinecone'])

    def test_all_fifteen_fixture_questions_match_linear_public_results(self):
        """Controlled first-five canonical evidence; not frozen retrieval scoring."""
        dataset = json.loads((ROOT/'evaluations/questions.json').read_text())
        chunks = tuple(c for d in load_documents(ROOT, mode='draft_review') for c in chunk_document(d).chunks)
        cat = catalog(chunks)
        seen = []
        for q in dataset['questions']:
            with self.subTest(question=q['question_id']):
                selected = tuple(c for c in chunks if any(c.document_id == e['document_id'] and
                    (c.section_id == e['section_id'] or c.section_id.startswith(e['section_id']+'.'))
                    for e in q['required_evidence']))[:5]
                self.assertTrue(selected)
                hits = tuple(hit(c) for c in selected)
                limits = GenerationLimits(context_tokens=131072)
                p = g.prepare_answer(q['question'], hits, mode=q['retrieval_mode'], reference_date=AS_OF, limits=limits)
                body = authored_claims(p, q['expected_behavior'])
                graph, _, _, _ = self.make(cat=cat, first=FixtureStore(hits), body=body)
                linear = qa.QAService(FixtureStore(hits), None, FakeLLM(body), corpus_fingerprint='fixture')
                options = dict(mode=q['retrieval_mode'], reference_date=AS_OF, limits=limits)
                actual = graph.ask(q['question'], **options)
                expected = linear.ask(q['question'], **options)
                self.assertTrue(actual.accepted, actual.public)
                self.assertEqual(actual.public, expected.public)
                self.assertEqual(actual.public.coverage.status, 'not_assessed')
                seen.append(q['question_id'])
        self.assertEqual(len(set(seen)), 15)

    def test_linear_and_ui_imports_do_not_require_optional_graph_packages(self):
        code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('langgraph','langchain_core','langsmith'):
            raise ImportError('optional graph dependency unavailable')
sys.meta_path.insert(0,Block())
from northstar_atlas import qa, answers
from northstar_atlas.ui import runtime
assert 'northstar_atlas.orchestration' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retry_enabled_provider_rejected_before_factory_backend_open(self):
        provider = FakeLLM(); provider._attempts = 2
        with patch.object(self.o, 'load_catalog', return_value=self.cat), patch.object(self.o, 'open_retriever') as opened:
            with self.assertRaises(qa.QAConfigurationError):
                with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, provider=provider): self.fail('opened')
        opened.assert_not_called()

    def test_invalid_authored_clarification_is_configuration_error_before_io(self):
        service, first, second, provider = self.make()
        hint = self.o.CallerReviewedClarification('Who?', 'draft_review', '', ())
        self.error(self.ask(service, clarification=hint), 'configuration_failed')
        self.assertEqual(first.calls + second.calls + provider.calls, [])

    def test_recursion_failure_after_search_retains_safe_attempt_accounting(self):
        service, first, _, provider = self.make()
        invoke = service._graph.invoke
        def exhausted(*args, **kwargs):
            kwargs['config'] = {**kwargs['config'], 'recursion_limit':2}
            return invoke(*args, **kwargs)
        with patch.object(service._graph, 'invoke', side_effect=exhausted):
            self.error(self.ask(service), 'internal_error')
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(provider.calls, [])
        self.assertEqual(service.last_trace.retrieval_attempts, 1)
        self.assertEqual(service.last_trace.nodes, ('router', 'retriever', 'validator'))

    def test_interrupted_setup_closes_owned_provider_and_preserves_interrupt(self):
        from northstar_atlas import nebius
        closed = []
        class OwnedProvider:
            # Like NebiusProvider, cleanup requires __exit__; no generator GC
            # finalizer can accidentally hide missing ExitStack cleanup.
            def __enter__(self): return FakeLLM()
            def __exit__(self, *args): closed.append('provider')
        interruption = KeyboardInterrupt()
        with patch.object(self.o, 'load_catalog', return_value=self.cat), \
             patch.object(nebius, 'load_settings', return_value=SimpleNamespace(model=g.MODEL_ID)), \
             patch.object(nebius, 'NebiusProvider', return_value=OwnedProvider()), \
             patch.object(self.o, 'open_retriever', side_effect=interruption):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with self.o.open_graph_qa_service(ROOT, snapshot_id='a'*64, allow_paid=True): self.fail('opened')
            self.assertEqual(closed, ['provider'])
            self.assertIs(caught.exception, interruption)


if __name__ == '__main__': unittest.main()
