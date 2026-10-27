"""Phase B semantic-check graph tests use authored responses and fake providers only."""
from dataclasses import asdict
import json
import unittest
from unittest.mock import patch

from test_context import AS_OF, ROOT, chunk, hit
from test_bm25 import catalog
from test_generation_v2 import payload
from test_qa import FakeLLM, FixtureStore, authored_claims
from test_orchestration_assessment import route
from northstar_atlas import generation as g, orchestration as o, qa
from northstar_atlas.grounding import EvidenceObligation
from northstar_atlas.orchestration_assessment import SemanticContextError
from northstar_atlas.grounding import AbstentionRequirements
from northstar_atlas.evidence import format_citation
from northstar_atlas.llm import GenerationResponse, TokenUsage


class ScopedStore(FixtureStore):
    def search(self, question, embedder, *, department_scope=None, **kwargs):
        return super().search(question, embedder, department_scope=department_scope, **kwargs)


class PhaseBSemanticGraphTests(unittest.TestCase):
    def setUp(self):
        self.source = chunk()
        self.hits = (hit(self.source),)
        self.cat = catalog([self.source])

    @staticmethod
    def semantic_reply(data, result='supported'):
        rows = []
        for item in data['items']:
            present = item['assertion_required'] or (
                result == 'public_field_unsupported'
                and item['field'] in ('clarification_question', 'missing_information'))
            support = ('unsupported' if result in ('unsupported', 'mixed', 'public_field_unsupported')
                       else 'supported') if present else 'not_applicable'
            rows.append({
                'item_id': item['item_id'],
                'assertion': 'present' if present else 'none',
                'support': support,
                'authority': 'misstated' if result == 'authority_misstated' and present else
                             'unclear' if result == 'mixed' and present else 'aligned' if present else 'not_applicable',
                'conditions': 'missing' if result == 'conditions_missing' and present else
                              'complete' if present else 'not_applicable',
            })
        overall = {'question_coverage': 'complete', 'gap_handling': 'accurate'}
        if result == 'incomplete': overall['question_coverage'] = 'incomplete'
        if result == 'unclear': overall['question_coverage'] = 'unclear'
        if result == 'gap_misrepresented': overall['gap_handling'] = 'misrepresented'
        return {'schema_version': 'semantic-check-v1', 'assessments': rows,
                'answer_assessment': overall}

    def make(self, *, first=None, second=None, judgments=(), semantics=(), body=None,
             plan=('dense', 'bm25'), provider=None, cat=None):
        first = ScopedStore(self.hits) if first is None else first
        second = ScopedStore(self.hits) if second is None else second
        grade_count = 0
        semantic_count = 0

        def reply(request):
            nonlocal grade_count, semantic_count
            data = json.loads(request.user)
            if 'chunks' in data:
                choice = judgments[grade_count] if grade_count < len(judgments) else 'relevant'
                grade_count += 1
                if callable(choice):
                    return choice(data)
                return {'schema_version': 'relevance-v1', 'judgments': [
                    {'chunk_id': row['chunk_id'], 'relevance': choice} for row in data['chunks']]}
            if set(data) == {'question'}:
                return route()
            if 'candidate_text' in data:
                outcome = semantics[semantic_count] if semantic_count < len(semantics) else 'supported'
                semantic_count += 1
                if callable(outcome):
                    return outcome(data)
                if outcome == 'malformed':
                    return {'schema_version': 'semantic-check-v1', 'assessments': [],
                            'answer_assessment': {'question_coverage': 'complete', 'gap_handling': 'accurate'}}
                return self.semantic_reply(data, outcome)
            if callable(body):
                return body(request)
            if body is not None:
                return body
            prepared = g.prepare_answer('Who?', self.hits, mode='draft_review', reference_date=AS_OF)
            return authored_claims(prepared)

        provider = provider or FakeLLM(reply)
        service = o.GraphQAService({'dense': first, 'bm25': second, 'hybrid': first}, None, provider,
            catalog=cat or self.cat, strategy_plan=plan, profile='llm_routing_grading_semantic_v1')
        return service, first, second, provider

    def ask(self, service, **kwargs):
        return service.ask('Who?', mode='draft_review', reference_date=AS_OF, **kwargs)

    def assert_failure(self, result, code):
        self.assertIsInstance(result, qa.QAFailure)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, code)
        self.assertNotIn('SECRET_REJECTED', json.dumps(asdict(result), default=str))
        self.assertFalse(hasattr(result.public, 'text'))

    def test_supported_complete_answer_is_published_after_semantic_check(self):
        service, _, _, provider = self.make()
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.status, 'not_assessed')
        self.assertIn('(proposed)', result.public.text)
        self.assertEqual([call.purpose for call in service.last_trace.calls],
                         ['router', 'grader', 'generator', 'semantic_checker'])
        self.assertEqual(service.last_trace.nodes,
                         ('router', 'retriever', 'assessor', 'generator', 'semantic_checker', 'validator'))
        self.assertEqual(len(provider.calls), 4)

    def test_supported_statements_but_incomplete_answer_regenerate_then_pass(self):
        service, _, _, provider = self.make(semantics=('incomplete', 'supported'))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(len(provider.calls), 6)
        self.assertEqual([call.purpose for call in service.last_trace.calls],
                         ['router', 'grader', 'generator', 'semantic_checker', 'generator', 'semantic_checker'])
        self.assertIn('incomplete_answer', provider.calls[4].system)

    def test_alternate_retrieval_plus_regeneration_never_exceeds_seven_calls(self):
        service, _, _, provider = self.make(judgments=('uncertain', 'relevant'),
                                               semantics=('conditions_missing', 'supported'))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertEqual(len(provider.calls), 7)
        self.assertEqual(len(service.last_trace.calls), 7)
        self.assertLessEqual(sum(call.input_token_bound for call in service.last_trace.calls), 184064)
        self.assertLessEqual(sum(call.output_token_limit for call in service.last_trace.calls), 20736)
        self.assertLessEqual(sum(call.input_token_bound + call.output_token_limit
                                 for call in service.last_trace.calls), 204800)
        self.assertEqual([call.purpose for call in service.last_trace.calls].count('grader'), 2)
        self.assertEqual(service.last_trace.calls[-1].purpose, 'semantic_checker')

    def test_proposed_authority_misstatement_triggers_one_regeneration_and_preserves_status(self):
        service, _, _, provider = self.make(semantics=('authority_misstated', 'supported'))
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertIn('(proposed)', result.public.text)
        self.assertIn('authority_misstatement', provider.calls[4].system)
        self.assertEqual([call.purpose for call in service.last_trace.calls].count('generator'), 2)

    def test_second_negative_and_mixed_negative_unclear_are_withheld_without_more_generation(self):
        for semantics in (('incomplete', 'unsupported'), ('mixed',)):
            service, _, _, provider = self.make(semantics=semantics)
            self.assert_failure(self.ask(service), 'response_rejected')
            self.assertEqual([call.purpose for call in service.last_trace.calls].count('generator'),
                             2 if len(semantics) == 2 else 1)
            self.assertEqual(len(provider.calls), 6 if len(semantics) == 2 else 4)

    def test_malformed_semantic_output_and_checker_provider_failure_fail_closed(self):
        service, _, _, provider = self.make(semantics=('malformed',))
        self.assert_failure(self.ask(service), 'response_rejected')
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(service.last_trace.calls[-1].status, 'response_rejected')

        class FailingChecker(FakeLLM):
            def generate(inner, request):
                if 'candidate_text' in json.loads(request.user):
                    inner.calls.append(request)
                    raise TimeoutError('SECRET_REJECTED')
                return super(FailingChecker, inner).generate(request)

        # Reuse the ordinary deterministic fixtures for the non-checker stages.
        base, _, _, base_provider = self.make()
        provider = FailingChecker()
        provider.body = base_provider.body
        service, _, _, _ = self.make(provider=provider)
        self.assert_failure(self.ask(service), 'provider_failed')
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(service.last_trace.calls[-1].status, 'provider_failed')

    def test_incomplete_checker_transport_response_is_rejected_without_regeneration(self):
        class TruncatedChecker(FakeLLM):
            def generate(inner, request):
                if 'candidate_text' in json.loads(request.user):
                    inner.calls.append(request)
                    return GenerationResponse('{}', request.model, 'length',
                                              TokenUsage(100, 50, 150), .01, 1)
                return super(TruncatedChecker, inner).generate(request)

        provider = TruncatedChecker()
        base, _, _, base_provider = self.make()
        provider.body = base_provider.body
        service, _, _, _ = self.make(provider=provider)
        self.assert_failure(self.ask(service), 'response_rejected')
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(service.last_trace.calls[-1].status, 'response_rejected')

    def test_deterministic_generation_failure_prevents_semantic_call_or_regeneration(self):
        service, _, _, provider = self.make(body='SECRET_REJECTED', semantics=('incomplete',))
        self.assert_failure(self.ask(service), 'response_rejected')
        self.assertEqual([call.purpose for call in service.last_trace.calls],
                         ['router', 'grader', 'generator'])
        self.assertEqual(len(provider.calls), 3)

    def test_invalid_citation_is_rejected_before_any_semantic_call(self):
        prepared = g.prepare_answer('Who?', self.hits, mode='draft_review', reference_date=AS_OF)
        body = authored_claims(prepared)
        body['main_answer']['claims'][0]['support'][0]['citation'] = 'invented citation'
        service, _, _, provider = self.make(body=body, semantics=('incomplete',))
        self.assert_failure(self.ask(service), 'response_rejected')
        self.assertEqual([call.purpose for call in service.last_trace.calls],
                         ['router', 'grader', 'generator'])
        self.assertEqual(len(provider.calls), 3)

    def test_context_budget_failure_happens_before_checker_io(self):
        service, _, _, provider = self.make()
        with patch.object(o, 'prepare_semantic_check', side_effect=SemanticContextError('too large')):
            self.assert_failure(self.ask(service), 'context_failed')
        self.assertEqual(len(provider.calls), 3)
        self.assertNotIn('semantic_checker', [call.purpose for call in service.last_trace.calls])

    def test_generated_clarification_and_abstention_text_are_in_semantic_inventory(self):
        for kind in ('clarify', 'abstain'):
            body = payload(kind=kind)
            body['main_answer']['claims'] = []
            service, _, _, provider = self.make(body=body)
            result = self.ask(service)
            self.assertTrue(result.accepted)
            self.assertEqual(result.public.kind, kind)
            check_data = json.loads(provider.calls[-1].user)
            fields = [item['field'] for item in check_data['items']]
            if kind == 'clarify':
                self.assertIn('clarification_question', fields)
                self.assertIn('missing_information', fields)
            else:
                self.assertEqual(fields, ['missing_information'])

    def test_generated_policy_assertion_in_clarification_is_not_treated_as_neutral(self):
        body = payload(kind='clarify')
        body['main_answer']['claims'] = []
        body['missing_information'] = ['The policy requires manager approval.']
        service, _, _, provider = self.make(body=body,
            semantics=('public_field_unsupported', 'public_field_unsupported'))
        self.assert_failure(self.ask(service), 'response_rejected')
        self.assertEqual([call.purpose for call in service.last_trace.calls].count('generator'), 2)
        self.assertTrue(all(any(item['field'] == 'missing_information' and
                                item['text'] == 'The policy requires manager approval.'
                                for item in json.loads(call.user)['items'])
                            for call in provider.calls if 'candidate_text' in json.loads(call.user)))

    def test_generated_clarification_or_abstention_cannot_skip_semantic_failure(self):
        for kind in ('clarify', 'abstain'):
            body = payload(kind=kind)
            body['main_answer']['claims'] = []
            service, _, _, provider = self.make(body=body, semantics=('incomplete', 'incomplete'))
            self.assert_failure(self.ask(service), 'response_rejected')
            self.assertEqual([call.purpose for call in service.last_trace.calls].count('generator'), 2)
            checks = [json.loads(call.user) for call in provider.calls
                      if 'candidate_text' in json.loads(call.user)]
            self.assertEqual(len(checks), 2)
            self.assertTrue(all(check['items'] for check in checks))

    def test_empty_evidence_abstention_is_trusted_template_with_no_check_call(self):
        service, first, second, provider = self.make(first=ScopedStore(), second=ScopedStore())
        result = self.ask(service)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual([call.purpose for call in service.last_trace.calls], ['router'])
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(len(provider.calls), 1)

    def test_phase_b_approved_only_rejects_draft_before_generation(self):
        service, first, second, provider = self.make()
        self.assert_failure(service.ask('Who?', mode='approved_only', reference_date=AS_OF),
                           'context_failed')
        self.assertEqual(first.calls[0][2]['mode'], 'approved_only')
        self.assertEqual(second.calls, [])
        self.assertEqual(len(provider.calls), 1)  # The Phase A router ran before retrieval.
        self.assertEqual([call.purpose for call in service.last_trace.calls], ['router'])

    def test_trusted_application_clarification_skips_all_model_calls(self):
        service, first, second, provider = self.make()
        hint = o.CallerReviewedClarification('Who?', 'draft_review', 'Which environment?',
                                             ('Access environment.',))
        result = self.ask(service, clarification=hint)
        self.assertEqual(result.public.kind, 'clarify')
        self.assertEqual(first.calls + second.calls + provider.calls, [])
        self.assertEqual(service.last_trace.calls, ())

    def test_reviewed_obligation_remains_required_and_semantically_checked(self):
        package = g.prepare_answer('Who?', self.hits, mode='draft_review', reference_date=AS_OF).context
        obligation = EvidenceObligation('required', package.groups[0].citation,
                                        self.source.text.strip(), 'extractive')
        service, _, _, provider = self.make(body=payload((obligation,)))
        result = self.ask(service, obligations=(obligation,))
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.coverage.extractive_ids, ('required',))
        self.assertEqual([item['field'] for item in json.loads(provider.calls[-1].user)['items']],
                         ['obligation_statement'])

    def test_caller_reviewed_evidence_backed_abstention_still_runs_checker(self):
        package = g.prepare_answer('What is the approved duration?', self.hits,
                                   mode='draft_review', reference_date=AS_OF).context
        obligation = EvidenceObligation('no_period_established', package.groups[0].citation,
                                        self.source.text.strip(), 'extractive')
        requirements = AbstentionRequirements((obligation,),
            ('The corpus does not establish an approved duration.',), 'documented_gap')
        body = {'kind': 'abstain', 'main_answer': {'claims': [], 'obligations': {
            'no_period_established': {'statement': obligation.source_quote,
                'support': [{'citation': format_citation(obligation.citation),
                             'quote': obligation.source_quote}], 'unresolved_reason': None}}},
            'missing_information': list(requirements.missing_information),
            'clarification_question': None, 'reason': 'insufficient_evidence'}
        service, _, _, provider = self.make(body=body)
        result = self.ask(service, abstention=requirements)
        self.assertTrue(result.accepted)
        self.assertEqual(result.public.kind, 'abstain')
        self.assertEqual(result.public.presentation.abstention_basis, 'documented_gap')
        self.assertIn('(proposed)', result.public.text)
        self.assertEqual([call.purpose for call in service.last_trace.calls],
                         ['router', 'grader', 'generator', 'semantic_checker'])

    def test_phase_a_and_b_profiles_remain_explicit_and_paid_factory_stays_closed(self):
        with self.assertRaises(qa.QAConfigurationError):
            o.GraphQAService({'dense': ScopedStore(self.hits)}, None, FakeLLM('{}'), catalog=self.cat,
                             strategy_plan=('dense',), profile='unknown')
        with patch.object(o, 'load_catalog', side_effect=AssertionError('catalog accessed')):
            with self.assertRaises(qa.QAConfigurationError):
                with o.open_graph_qa_service(ROOT, snapshot_id='unused',
                        profile='llm_routing_grading_semantic_v1'):
                    pass


if __name__ == '__main__':
    unittest.main()
