"""Semantic graph with real Pinecone codecs and inert in-memory HTTP; no cloud IO."""
from contextlib import contextmanager
from dataclasses import replace
import json
import socket
import unittest
from unittest.mock import patch

import test_pinecone_store as pinecone_fixture
from test_context import ROOT, AS_OF, MODEL, chunk, family
from test_qa import FakeLLM, authored_claims
from test_orchestration_assessment import route
import test_orchestration_semantic as semantic_fixtures
from northstar_atlas import generation, orchestration
from northstar_atlas.embeddings import embed_chunks
from northstar_atlas.vector_store.catalog import ChunkCatalog
from northstar_atlas.vector_store.records import IndexConfig
from northstar_atlas.vector_store.preparation import freeze_snapshot


class PineconeSemanticGraphTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        pinecone_fixture.PineconeTests.setUpClass()
        self.f = pinecone_fixture.PineconeTests()
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.original = chunk('Only within the approved access scope.\nRequired conditions remain applicable.\n')
        self.parts = family(self.original, [len(self.original.text.splitlines(keepends=True)[0])])
        self.f.provider.model = MODEL
        self.f.config = IndexConfig(MODEL, 256)
        self.f.current = 'fixture'
        self.f.snapshot = freeze_snapshot(embed_chunks(self.parts, self.f.provider), self.f.config, self.f.current)
        self.f.store = self.f.make_store()
        self.f.store.synchronize()  # Inert fixture only; never a provider or existing publication.
        self.cat = ChunkCatalog([self.original], self.parts, self.f.config, self.f.current)
        self.hits = self.f.search(mode='draft_review', top_k=1)
        self.f.http.calls.clear()
        self.question = 'Who approves access?'

    def provider(self, *, judgments=(), semantics=(), malformed_grading=False):
        counts = {'grader': 0, 'checker': 0}
        prepared = generation.prepare_answer(self.question, self.hits, mode='draft_review', reference_date=AS_OF)
        def reply(request):
            data = json.loads(request.user)
            if set(data) == {'question'}:
                return route(['it_operations'])
            if 'chunks' in data:
                i = counts['grader']; counts['grader'] += 1
                choice = judgments[i] if i < len(judgments) else 'relevant'
                rows = [{'chunk_id': c['chunk_id'], 'relevance': choice} for c in data['chunks']]
                return {'schema_version': 'relevance-v1', 'judgments': rows[:1] if malformed_grading else rows}
            if 'candidate_text' in data:
                i = counts['checker']; counts['checker'] += 1
                choice = semantics[i] if i < len(semantics) else 'supported'
                return semantic_fixtures.PhaseBSemanticGraphTests.semantic_reply(data, choice)
            return authored_claims(prepared)
        return FakeLLM(reply)

    @contextmanager
    def service(self, provider, plan=('dense', 'bm25')):
        opened = []
        @contextmanager
        def retriever(root, backend, *, snapshot_id):
            self.assertEqual(backend, 'pinecone')
            self.assertEqual(snapshot_id, self.f.snapshot.snapshot_id)
            opened.append(True)
            yield self.f.store, self.f.provider, self.f.current
        with patch.object(orchestration, 'load_catalog', return_value=self.cat), \
             patch.object(orchestration, 'open_retriever', side_effect=retriever):
            with orchestration.open_graph_qa_service(ROOT, snapshot_id=self.f.snapshot.snapshot_id,
                    backend='pinecone', profile=orchestration.SEMANTIC_PROFILE, provider=provider,
                    strategy_plan=plan) as service:
                yield service
            self.assertEqual(opened, [True])

    def ask(self, service, mode='draft_review'):
        return service.ask(self.question, mode=mode, reference_date=AS_OF, top_k=1)

    def test_all_nodes_use_category_scope_and_grade_each_required_sibling(self):
        provider = self.provider()
        with self.service(provider) as service:
            result = self.ask(service)
            self.assertTrue(result.accepted)
            self.assertEqual(service.last_trace.nodes,
                ('router', 'retriever', 'assessor', 'generator', 'semantic_checker', 'validator'))
            self.assertEqual([c.purpose for c in service.last_trace.calls],
                             ['router', 'grader', 'generator', 'semantic_checker'])
        self.assertEqual(result.diagnostics.backend, 'pinecone')
        self.assertEqual(result.public.coverage.status, 'not_assessed')
        self.assertIn('(proposed)', result.public.text)
        self.assertEqual(result.public.citations[0].content_hash, self.original.content_hash)
        rows = json.loads(provider.calls[1].user)['chunks']
        self.assertEqual({c['chunk_id'] for c in rows}, {c.chunk_id for c in self.parts})
        search = next(body for path, body, _ in self.f.http.calls if path.endswith('/search'))
        self.assertIn({'department': {'$in': ['IT Operations']}}, search['filter']['$and'])
        self.assertFalse(any(path.endswith('/upsert') for path, _, _ in self.f.http.calls))

    def test_alternate_hybrid_and_regeneration_preserve_seven_call_bound(self):
        provider = self.provider(judgments=('uncertain', 'relevant'), semantics=('incomplete', 'supported'))
        with self.service(provider, plan=('dense', 'hybrid')) as service:
            self.assertTrue(self.ask(service).accepted)
            trace = service.last_trace
            self.assertEqual(trace.retrieval_attempts, 2)
            self.assertEqual(len(trace.calls), 7)
            self.assertLessEqual(sum(c.input_token_bound + c.output_token_limit for c in trace.calls), 204800)
        searches = [body for path, body, _ in self.f.http.calls if path.endswith('/search')]
        self.assertEqual(len(searches), 2)
        self.assertTrue(any('department' in c for c in searches[0]['filter']['$and']))
        self.assertFalse(any('department' in c for c in searches[1]['filter']['$and']))

    def test_approved_only_never_promotes_draft_evidence(self):
        provider = self.provider()
        with self.service(provider) as service:
            result = self.ask(service, mode='approved_only')
            self.assertTrue(result.accepted)
            self.assertEqual(result.public.kind, 'abstain')
            self.assertFalse(result.public.citations)
            self.assertEqual(service.last_trace.retrieval_attempts, 2)
        self.assertEqual(len(provider.calls), 1)  # Router only, trusted no-evidence terminal.

    def test_missing_sibling_judgment_is_rejected_before_generation(self):
        provider = self.provider(malformed_grading=True)
        with self.service(provider) as service:
            result = self.ask(service)
            self.assertFalse(result.accepted)
            self.assertEqual(result.public.code, 'response_rejected')
        self.assertEqual(len(provider.calls), 2)
        self.assertFalse(hasattr(result.public, 'text'))

    def test_missing_remote_sibling_fails_closed_before_grading(self):
        del self.f.http.spaces[self.f.store.namespace][self.parts[1].chunk_id]
        provider = self.provider()
        with self.service(provider) as service:
            result = self.ask(service)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, 'retrieval_failed')
        self.assertEqual(len(provider.calls), 1)

    def test_forged_source_cannot_reach_generation(self):
        provider = self.provider()
        forged = replace(self.hits[0], chunk=replace(self.hits[0].chunk, content_hash='b' * 64))
        with self.service(provider) as service, patch.object(self.f.store, 'search', return_value=(forged,)):
            result = self.ask(service)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, 'context_failed')
        self.assertEqual(len(provider.calls), 1)

    def test_negative_semantic_result_exhaustion_withholds_candidate(self):
        provider = self.provider(semantics=('incomplete', 'unsupported'))
        with self.service(provider) as service:
            result = self.ask(service)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, 'response_rejected')
        self.assertFalse(hasattr(result.public, 'text'))
        self.assertEqual(len(provider.calls), 6)

    def test_reservation_for_another_backend_rejects_before_routing(self):
        # Durable permit mismatch is covered by the live-wiring tests; the
        # shared service also checks before invoking any graph node.
        from unittest.mock import Mock
        from northstar_atlas.live_budget import BudgetError, WorkflowReservation
        provider = self.provider()
        with self.service(provider) as service:
            service._budget = Mock()
            permit = Mock(spec=WorkflowReservation)
            permit.check_request.side_effect = BudgetError()
            result = service.ask(self.question, mode='draft_review', reference_date=AS_OF, reservation=permit)
        self.assertFalse(result.accepted)
        self.assertEqual(result.public.code, 'configuration_failed')
        self.assertFalse(provider.calls)

