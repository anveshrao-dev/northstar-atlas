"""Real local retrieval in temporary checkouts with fake query/document vectors.

No production index is changed and no model/provider is contacted.
"""
from dataclasses import replace
from datetime import date
import json
import os
import socket
import unittest
from unittest.mock import patch

from checkout_fixture import checkout
from test_vector_store import FakeProvider
from northstar_atlas.embeddings import embed_chunks
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.ui.runtime import ResourceOwner, Submission
from northstar_atlas.vector_store import ChromaStore, IndexConfig
from northstar_atlas.vector_store.preparation import corpus_snapshot


class LocalPolicySearchTests(unittest.TestCase):
    def setUp(self):
        self.root = checkout(self)
        self.enterContext(patch.dict(os.environ, NORTHSTAR_PROJECT_ROOT=str(self.root),
            NORTHSTAR_CONTROLLED_LIVE='1', NORTHSTAR_UI_ENABLE_LIVE='0'))
        self.enterContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network blocked')))
        self.enterContext(patch.object(socket.socket, 'connect_ex', side_effect=AssertionError('network blocked')))
        self.enterContext(patch('northstar_atlas.nebius.NebiusProvider', side_effect=AssertionError('no inference')))
        self.enterContext(patch('northstar_atlas.live_budget.configured_budget', side_effect=AssertionError('no ledger')))
        self.enterContext(patch('northstar_atlas.vector_store.pinecone_cloud.load_settings',
                                side_effect=AssertionError('no hosted retrieval')))
        self.provider = FakeProvider()
        self.model_open = self.enterContext(patch('northstar_atlas.local_embeddings.MiniLMProvider',
                                                 return_value=self.provider))
        documents, self.fingerprint = corpus_snapshot(self.root)
        self.chunks = tuple(c for doc in documents[:2] for c in chunk_document(doc).chunks[:6])
        self.store = ChromaStore(self.root, IndexConfig(self.provider.model, 256))
        self.store.synchronize(embed_chunks(self.chunks, self.provider), corpus_fingerprint=self.fingerprint)
        self.owner = ResourceOwner(self.root)
        self.addCleanup(self.owner.close)

    def request(self, ident='one', **kwargs):
        return replace(Submission(ident, 'Who approves access for a transfer?', 'approved_only',
                                  'chroma', date(2026, 10, 29), retrieval_only=True), **kwargs)

    def test_real_store_top_five_exact_passages_status_and_reference_date(self):
        before = self.store.status()
        view = self.owner.submit(self.request(), consent=False)
        self.assertEqual(view.kind, 'evidence'); self.assertEqual(len(view.passages), 5)
        self.assertTrue(all(h.chunk.status == APPROVED and h.chunk.document_id == 'HR-001'
                            and h.chunk in self.chunks for h in view.passages))
        self.assertEqual([h.cosine_distance for h in view.passages],
                         sorted(h.cosine_distance for h in view.passages))
        draft = self.owner.submit(self.request('draft', mode='draft_review'), consent=False)
        self.assertEqual(draft.kind, 'evidence'); self.assertEqual(len(draft.passages), 5)
        self.assertTrue(all(h.chunk in self.chunks for h in draft.passages))
        early = self.owner.submit(self.request('early', reference_date=date(2000, 1, 1)), consent=False)
        self.assertEqual(early.kind, 'evidence'); self.assertFalse(early.passages)
        self.assertEqual(self.model_open.call_count, 1); self.assertEqual(self.owner.attempts, 0)
        self.assertEqual(self.store.status(), before)

    def test_modified_source_is_rejected_on_cached_resource_reuse(self):
        self.assertEqual(self.owner.submit(self.request(), consent=False).kind, 'evidence')
        source = self.root/'data/policies/HR-001.md'
        source.write_bytes(source.read_bytes()+b'\n')  # Synthetic checkout only.
        result = self.owner.submit(self.request('changed'), consent=False)
        self.assertEqual(result.code, 'search_failed'); self.assertFalse(result.passages)

    def test_incomplete_publication_and_corrupt_record_never_display_partial_hits(self):
        path = self.store.state_path
        original = path.read_bytes()
        for change in ({'state':'incomplete'}, {'record_digest':'0'*64}, {'corpus_fingerprint':'wrong'}):
            state = json.loads(original); state.update(change); path.write_text(json.dumps(state))
            result = self.owner.submit(self.request(str(len(change))+next(iter(change))), consent=False)
            self.assertEqual(result.code, 'search_failed'); self.assertFalse(result.passages)

    def test_missing_cached_model_is_safe_without_downloading_or_opening_budget(self):
        self.model_open.side_effect = ValueError('private model path')
        result = self.owner.submit(self.request(), consent=False)
        self.assertEqual(result.code, 'search_failed'); self.assertNotIn('private model path', result.text)
        self.assertEqual(self.model_open.call_count, 1)


if __name__ == '__main__':
    unittest.main()
