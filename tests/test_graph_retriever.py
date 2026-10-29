"""Exact graph snapshot binding at the shared, fully offline Pinecone opener."""
from dataclasses import replace
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_bm25 import catalog
from test_context import MODEL, ROOT, chunk
from test_qa import FakeLLM
from northstar_atlas import qa
from northstar_atlas.embeddings import EmbeddedChunk
from northstar_atlas.generation import GenerationError
from northstar_atlas.vector_store import catalog as catalog_module
from northstar_atlas.vector_store import pinecone_cloud as cloud
from northstar_atlas.vector_store import pinecone_workflow as workflow
from northstar_atlas.vector_store.preparation import freeze_snapshot


class Resource:
    def __init__(self, name, events):
        self.name, self.events, self._http = name, events, object()

    def __enter__(self):
        self.events.append(self.name + ':open')
        return self

    def __exit__(self, *_):
        self.events.append(self.name + ':close')


class DenseStore:
    def __init__(self, snapshot, events):
        self.snapshot, self.events = snapshot, events

    def audit(self):
        self.events.append('audit')

    def search(self, question, provider, *, department_scope=None, **options):
        raise AssertionError('factory setup must not search')

    def synchronize(self):
        raise AssertionError('retrieval must not reindex')


class GraphRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(socket.socket, 'connect',
                                      side_effect=AssertionError('network forbidden')))
        self.events = []
        self.cat = catalog([chunk()])
        self.snapshot = freeze_snapshot(
            [EmbeddedChunk(self.cat.chunks[0], (1., 0., 0.), MODEL)], self.cat.config, 'fixture')
        self.other_snapshot = freeze_snapshot(
            [replace(self.snapshot.records[0], vector=(0., 1., 0.))], self.cat.config, 'fixture')
        self.embedder = SimpleNamespace(model=MODEL, max_input_tokens=256)
        self.store = DenseStore(self.snapshot, self.events)
        self.client = Resource('client', self.events)
        self.index = Resource('index', self.events)
        self.target, self.settings = object(), object()
        self.model = self.enterContext(patch('northstar_atlas.local_embeddings.MiniLMProvider',
                                            return_value=self.embedder))
        self.enterContext(patch('northstar_atlas.vector_store.corpus_snapshot', return_value=([], 'fixture')))
        self.enterContext(patch.object(cloud, 'load_settings', return_value=self.settings))
        self.confirm = self.enterContext(patch.object(cloud, 'check_confirmation'))
        self.sdk = self.enterContext(patch.object(cloud, 'sdk_client', return_value=self.client))
        self.connect = self.enterContext(patch.object(workflow, 'connect_target', return_value=(self.target, self.index)))
        self.active = self.enterContext(patch.object(workflow, 'active_snapshot', return_value=self.snapshot))
        self.open_store = self.enterContext(patch.object(workflow, 'open_store', return_value=self.store))
        self.enterContext(patch.object(workflow, 'prepare_upload', side_effect=AssertionError('no preparation')))
        self.enterContext(patch.object(cloud, 'create_index', side_effect=AssertionError('no provisioning')))

    def test_matching_active_snapshot_opens_and_audits_existing_store_once(self):
        with qa.open_retriever(ROOT, 'pinecone', snapshot_id=self.snapshot.snapshot_id) as resources:
            self.assertEqual(resources, (self.store, self.embedder, 'fixture'))
            self.assertEqual(self.events, ['client:open', 'index:open', 'audit'])
        self.assertEqual(self.events, ['client:open', 'index:open', 'audit', 'index:close', 'client:close'])
        self.confirm.assert_called_once_with(ROOT, self.settings)
        self.active.assert_called_once_with(ROOT, self.target)
        self.open_store.assert_called_once_with(ROOT, self.snapshot, self.target, self.client, self.index)

    def test_same_corpus_and_model_with_different_snapshot_rejected_before_audit(self):
        self.active.return_value = self.other_snapshot
        with self.assertRaisesRegex(GenerationError, 'snapshot'):
            with qa.open_retriever(ROOT, 'pinecone', snapshot_id=self.snapshot.snapshot_id):
                self.fail('mismatched active snapshot opened')
        self.open_store.assert_not_called()
        self.assertEqual(self.events, ['client:open', 'index:open', 'index:close', 'client:close'])

    def test_unpinned_linear_dense_keeps_existing_active_generation_selection(self):
        self.active.return_value = self.other_snapshot
        with qa.open_retriever(ROOT, 'pinecone') as (store, _, _):
            self.assertIs(store, self.store)
        self.active.assert_called_once_with(ROOT, self.target)
        self.open_store.assert_called_once_with(ROOT, self.other_snapshot, self.target, self.client, self.index)

    def test_target_confirmation_failure_precedes_client_and_store_open(self):
        self.confirm.side_effect = cloud.CloudError('target not confirmed')
        with self.assertRaises(cloud.CloudError):
            with qa.open_retriever(ROOT, 'pinecone', snapshot_id=self.snapshot.snapshot_id):
                self.fail('unconfirmed target opened')
        self.sdk.assert_not_called()
        self.connect.assert_not_called()
        self.active.assert_not_called()
        self.open_store.assert_not_called()
        self.assertEqual(self.events, [])

    def test_hybrid_preserves_requested_snapshot_at_dense_open(self):
        self.active.return_value = self.other_snapshot
        with patch.object(catalog_module, 'load_catalog', return_value=self.cat):
            with self.assertRaisesRegex(GenerationError, 'snapshot'):
                with qa.open_retriever(ROOT, 'pinecone', strategy='hybrid',
                                       snapshot_id=self.snapshot.snapshot_id):
                    self.fail('hybrid discarded its requested snapshot')
        self.open_store.assert_not_called()
        self.assertNotIn('audit', self.events)

    def test_graph_factory_reuses_dense_store_for_hybrid_without_search_or_reindex(self):
        from northstar_atlas import orchestration
        provider = FakeLLM(error=AssertionError('setup must not generate'))
        with patch.object(orchestration, 'load_catalog', return_value=self.cat):
            with orchestration.open_graph_qa_service(ROOT, backend='pinecone',
                    snapshot_id=self.snapshot.snapshot_id, provider=provider,
                    profile=orchestration.SEMANTIC_PROFILE,
                    strategy_plan=('hybrid', 'dense')) as service:
                self.assertIs(service._stores['dense'], self.store)
                self.assertIs(service._stores['hybrid'].dense, self.store)
        self.assertEqual(provider.calls, [])
        self.model.assert_called_once_with(ROOT)
        self.open_store.assert_called_once()
        self.assertEqual(self.events, ['client:open', 'index:open', 'audit', 'index:close', 'client:close'])


if __name__ == '__main__':
    unittest.main()
