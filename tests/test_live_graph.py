"""Controlled graph wiring: synthetic ledgers, mocked HTTP, blocked sockets."""
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import socket
import unittest
from unittest.mock import Mock, patch

import httpx

from budget_fixture import BILLING, SCOPE, document, install
from checkout_fixture import ROOT, checkout
from northstar_atlas import live_graph
from northstar_atlas.live_budget import BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH, LINEAR, configured_budget
from northstar_atlas.llm import MODEL_ID
from northstar_atlas.nebius import NebiusProvider, NebiusSettings
from northstar_atlas.ui.runtime import ResourceOwner, Submission, LANGGRAPH_WORKFLOW, Conversation
from test_context import AS_OF
import test_orchestration_semantic as semantic_fixtures


class LiveGraphTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root),
            NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='1',
            NORTHSTAR_UI_ENABLE_GRAPH='1', NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED='1',
            NORTHSTAR_BUDGET_SCOPE=SCOPE, NORTHSTAR_BILLING_SCOPE=BILLING,
            NORTHSTAR_BUDGET_MAX_ATTEMPTS='8'))
        # Never inherit a real operator ledger while testing synthetic checkouts.
        os.environ.pop('NORTHSTAR_BUDGET_AUTHORITY_ROOT', None)
        os.environ.pop('NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED', None)
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.path = install(self.root, document(calls=8))
        self.access = self.enterContext(patch('northstar_atlas.live_preflight.require_private_preview'))
        self.budget = configured_budget(self.root)
        self.fixture = semantic_fixtures.PhaseBSemanticGraphTests(); self.fixture.setUp()
        self.created = []; self.sent = []; self.closed = []

    def request(self, ident='one'):
        return Submission(ident, 'Who?', 'draft_review', 'chroma', AS_OF, workflow=LANGGRAPH_WORKFLOW)

    def wire(self, *, judgments=(), semantics=(), failure=None):
        _, dense, _, fake = self.fixture.make(judgments=judgments, semantics=semantics)
        def send(req):
            flow = json.loads(self.path.read_text())['workflows'][-1]
            self.assertEqual(flow['status'], 'reserved')
            self.assertEqual(flow['calls'][-1]['status'], 'started')
            self.sent.append(flow['calls'][-1]['stage'])
            if failure:
                raise failure
            # The actual Nebius serializer/transport and deterministic graph validators run.
            body = json.loads(req.content)
            from northstar_atlas.llm import GenerationRequest
            response = fake.generate(GenerationRequest(body['messages'][0]['content'],
                body['messages'][1]['content'], body['response_format']['json_schema']['schema'],
                MODEL_ID, body['max_tokens'], 32768, 1000))
            return httpx.Response(200, json={'model': MODEL_ID,
                'choices': [{'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': response.text}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2, 'total_tokens': 12}})
        @contextmanager
        def provider(settings, **kwargs):
            flow = json.loads(self.path.read_text())['workflows'][-1]
            self.assertEqual((flow['profile'], flow['status'], flow['calls']), (GRAPH, 'reserved', []))
            self.created.append(kwargs)
            try:
                with NebiusProvider(settings, transport=httpx.MockTransport(send), **kwargs) as real:
                    yield real
            finally:
                self.closed.append(True)
        @contextmanager
        def retriever(root, backend):
            self.assertEqual(backend, 'chroma')
            yield dense, Mock(model=self.fixture.cat.config.model), self.fixture.cat.fingerprint
        self.enterContext(patch.object(live_graph, 'verify_graph_assets'))
        self.enterContext(patch('northstar_atlas.nebius.load_settings', return_value=NebiusSettings('synthetic-fixture')))
        self.enterContext(patch('northstar_atlas.nebius.NebiusProvider', side_effect=provider))
        self.enterContext(patch('northstar_atlas.orchestration.load_catalog', return_value=self.fixture.cat))
        self.enterContext(patch('northstar_atlas.orchestration.open_retriever', side_effect=retriever))
        # Exercise alternate retrieval without relying on ranking in this wiring test.
        self.enterContext(patch('northstar_atlas.vector_store.bm25.BM25Store', return_value=dense))

    def owner(self):
        owner = ResourceOwner(self.root, live_enabled=True, graph_enabled=True, budget=self.budget)
        self.addCleanup(owner.close)
        return owner

    def test_seven_stage_workflow_reserved_before_provider_and_closed_once(self):
        self.wire(judgments=('uncertain', 'relevant'), semantics=('incomplete', 'supported'))
        owner = self.owner()
        result = owner.submit(self.request(), consent=True)
        self.assertEqual(result.kind, 'answer')
        self.assertTrue(result.proposed)
        self.assertEqual(result.coverage, 'not_assessed')
        self.assertEqual(self.sent, ['router', 'grader', 'grader', 'generator', 'semantic_checker', 'generator', 'semantic_checker'])
        flow = json.loads(self.path.read_text())['workflows'][0]
        self.assertEqual(flow['status'], 'closed')
        self.assertEqual([(c['stage'], c['ordinal']) for c in flow['calls']],
            [('router', 0), ('grader', 1), ('grader', 2), ('generator', 0), ('semantic_checker', 1), ('generator', 1), ('semantic_checker', 2)])
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 7)
        self.assertEqual(self.created, [dict(allow_paid=True, max_attempts=1, requires_budget=True)])
        self.assertEqual(owner.submit(self.request(), consent=True).code, 'duplicate')
        owner.close(); self.assertEqual(len(self.closed), 1)

    def test_external_budget_authority_keeps_graph_and_assets_bound_to_application(self):
        authority = checkout(self)
        local_path = self.path; local_before = local_path.read_bytes()
        self.path = install(authority, document(calls=8))
        self.enterContext(patch.dict(os.environ,
            NORTHSTAR_BUDGET_AUTHORITY_ROOT=str(authority),
            NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED=str(authority)))
        self.budget = configured_budget(self.root)
        self.wire(judgments=('uncertain', 'relevant'), semantics=('incomplete', 'supported'))
        result = self.owner().submit(self.request(), consent=True)
        self.assertEqual(result.kind, 'answer')
        self.assertEqual(result.coverage, 'not_assessed')
        self.assertEqual(self.budget.root, self.root)
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 7)
        self.assertEqual(len(self.sent), 7)
        self.assertEqual(local_path.read_bytes(), local_before)
        self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['status'], 'closed')
        live_graph.verify_graph_assets.assert_called_with(self.root)

    def test_two_sessions_share_resources_not_invocation_handles_or_history(self):
        self.wire()
        owner = self.owner(); first = Conversation(); second = Conversation()
        first.queue(self.request()); first.run_pending(owner, consent=True)
        # Four calls spent; a second full seven-slot reservation cannot fit in eight.
        second.queue(self.request('two')); second.run_pending(owner, consent=True)
        self.assertEqual(first.history[0].view.kind, 'answer')
        self.assertEqual(second.history[0].view.code, 'budget_unavailable')
        self.assertEqual(len(self.created), 1)
        self.assertEqual(len(self.sent), 4)
        self.assertEqual(len(json.loads(self.path.read_text())['workflows']), 1)

    def test_cached_graph_uses_new_reservations_and_rechecks_access_each_time(self):
        self.path.write_text(json.dumps(document(calls=15)))
        self.budget = PersistentBudget(self.root, scope=SCOPE, billing_scope=BILLING,
            policy=BudgetPolicy(max_attempts=15), before_request=self.access)
        self.wire(); owner = self.owner()
        for ident in ('first-session', 'second-session'):
            self.assertEqual(owner.submit(self.request(ident), consent=True).kind, 'answer')
        self.assertEqual(len(self.created), 1)
        self.assertEqual(len(self.sent), 8)
        flows = json.loads(self.path.read_text())['workflows']
        self.assertEqual([f['id'] for f in flows], ['first-session', 'second-session'])
        self.assertEqual([len(f['calls']) for f in flows], [4, 4])
        self.assertEqual(self.access.call_count, 10)  # Each workflow + each transport.
        owner.close(); owner.close(); self.assertEqual(len(self.closed), 1)

    def test_pricing_expiring_during_access_check_prevents_provider_construction(self):
        self.wire()
        now = self.budget.clock()
        self.access.side_effect = lambda: setattr(self.budget, 'clock', lambda: now + timedelta(hours=2))
        result = self.owner().submit(self.request(), consent=True)
        self.assertEqual(result.kind, 'error'); self.assertFalse(self.created)

    def test_uncertain_transport_holds_full_envelope_and_withholds_answer(self):
        self.wire(failure=httpx.ReadTimeout('private transport detail'))
        result = self.owner().submit(self.request(), consent=True)
        self.assertEqual(result.kind, 'error'); self.assertFalse(result.citations)
        self.assertNotIn('private transport detail', result.text)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 7)
        self.assertEqual(json.loads(self.path.read_text())['workflows'][0]['status'], 'uncertain')

    def test_revoked_operator_flag_and_pinecone_prevent_reservation(self):
        self.wire(); owner = self.owner()
        with patch.dict(os.environ, NORTHSTAR_UI_ENABLE_GRAPH='0'):
            self.assertEqual(owner.submit(self.request(), consent=True).code, 'workflow_disabled')
        self.assertEqual(owner.submit(replace(self.request(), backend='pinecone'), consent=True).code,
                         'workflow_backend_unsupported')
        self.assertFalse(json.loads(self.path.read_text())['workflows']); self.assertFalse(self.created)

    def test_access_failure_prevents_credentials_and_construction(self):
        self.wire(); self.access.side_effect = ValueError('private access detail')
        with patch('northstar_atlas.nebius.load_settings', side_effect=AssertionError('credentials accessed')):
            result = self.owner().submit(self.request(), consent=True)
        self.assertEqual(result.kind, 'error'); self.assertFalse(self.created); self.assertFalse(self.sent)
        self.assertEqual(self.budget.inspect_history().attempts_consumed, 0)

    def test_corrupt_expired_or_insufficient_budget_prevents_all_setup(self):
        self.wire()
        for mode in ('corrupt', 'expired', 'insufficient'):
            with self.subTest(mode=mode):
                value = document(calls=8)
                if mode == 'corrupt': value = {}
                if mode == 'expired':
                    from datetime import datetime
                    price = value['pricing_revisions'][0]
                    price['valid_until'] = (datetime.fromisoformat(price['confirmed_at']) + timedelta(seconds=1)).isoformat()
                if mode == 'insufficient': value['max_attempts'] = 3
                self.path.write_text(json.dumps(value))
                result = self.owner().submit(self.request(mode), consent=True)
                self.assertEqual(result.code, 'budget_unavailable')
        self.access.assert_not_called(); self.assertFalse(self.created)

    def test_missing_wrong_or_closed_reservation_cannot_open_provider(self):
        self.wire()
        for reservation in (None, object()):
            with self.assertRaises(ValueError):
                with live_graph.open_live_graph_service(self.root, budget=self.budget, reservation=reservation): pass
        with self.budget.reserve(WorkflowRequest('linear', LINEAR, 'Who?', 'draft_review', 'chroma', AS_OF)) as permit:
            with self.assertRaises(ValueError):
                with live_graph.open_live_graph_service(self.root, budget=self.budget, reservation=permit): pass
        with self.budget.reserve(WorkflowRequest('closed', GRAPH, 'Who?', 'draft_review', 'chroma', AS_OF)) as permit: pass
        with self.assertRaises(ValueError):
            with live_graph.open_live_graph_service(self.root, budget=self.budget, reservation=permit): pass
        self.assertFalse(self.created); self.access.assert_not_called()

    def test_attempt_policy_is_explicit_strict_and_never_inferred_from_ledger(self):
        for text in ('', '0', '+8', '8.0', '08', ' 8', '10001'):
            with self.subTest(value=text), patch.dict(os.environ, NORTHSTAR_BUDGET_MAX_ATTEMPTS=text):
                with self.assertRaises(BudgetError): configured_budget(self.root)
        with patch.dict(os.environ):
            os.environ.pop('NORTHSTAR_BUDGET_MAX_ATTEMPTS')
            self.assertEqual(configured_budget(self.root).policy.max_attempts, 3)

    def test_graph_launcher_preflight_is_local_until_a_workflow_is_reserved(self):
        from contextlib import redirect_stdout
        import io
        from northstar_atlas import live_preflight
        with patch.object(Path, 'cwd', return_value=self.root), \
             patch.object(live_graph, 'verify_graph_assets'), \
             patch.object(live_preflight, 'require_secret_environment'), \
             patch('northstar_atlas.qa.open_retriever', side_effect=AssertionError('unexpected retrieval setup')), \
             patch('northstar_atlas.nebius.NebiusProvider', side_effect=AssertionError('unreserved provider construction')), \
             patch.dict(os.environ, VECTOR_STORE='chroma', NORTHSTAR_UI_ENABLE_LIVE='0'), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(live_preflight.main(), 0)
        self.access.assert_not_called()
        self.assertFalse(json.loads(self.path.read_text())['workflows'])

    def test_readiness_for_graph_requires_the_whole_envelope(self):
        self.budget.check_ready(profile=GRAPH)
        self.budget.policy = BudgetPolicy(max_attempts=3)
        self.path.write_text(json.dumps(document(calls=3)))
        self.budget.check_ready(profile=LINEAR)
        with self.assertRaises(BudgetError): self.budget.check_ready(profile=GRAPH)

    def test_asset_verification_rejects_missing_tampered_snapshot_and_stale_index(self):
        # Real frozen data verification; Chroma status is mocked to avoid opening a database.
        relative = '.local/pinecone/snapshots/' + live_graph.SNAPSHOT_ID + '.json'
        target = self.root / relative
        with self.assertRaises(ValueError): live_graph.verify_graph_assets(self.root)
        target.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / relative, target)
        from northstar_atlas.local_embeddings import model_snapshot, MODEL_FILES
        for name in MODEL_FILES:
            asset = model_snapshot(self.root) / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.write_text('presence-only fixture; never loaded as a model')
        # Official Hub snapshots link files to blobs within the project cache.
        asset = model_snapshot(self.root) / 'config.json'
        blob = asset.with_name('fixture-blob.json')
        asset.rename(blob)
        asset.symlink_to(blob)
        ready = dict(state='ready', count=189, actual_count=189, corpus_fingerprint=live_graph.CORPUS_FINGERPRINT)
        with patch('northstar_atlas.vector_store.store.ChromaStore.status', return_value=ready):
            live_graph.verify_graph_assets(self.root)
            for change in ({'state':'incomplete'}, {'actual_count':188}, {'corpus_fingerprint':'stale'}):
                with patch('northstar_atlas.vector_store.store.ChromaStore.status', return_value=ready | change):
                    with self.assertRaises(ValueError): live_graph.verify_graph_assets(self.root)
        asset.unlink(); asset.symlink_to(ROOT / 'pyproject.toml')
        with self.assertRaises(ValueError): live_graph.verify_graph_assets(self.root)
        asset.unlink(); asset.symlink_to(blob)
        target.write_bytes(target.read_bytes() + b' ')
        with self.assertRaises(ValueError): live_graph.verify_graph_assets(self.root)
