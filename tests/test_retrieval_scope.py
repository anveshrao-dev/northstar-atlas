"""Scoped retrieval over real local adapters and an inert Pinecone HTTP boundary."""
from dataclasses import replace
import inspect
import unittest

import test_pinecone_store as pinecone_fixture
import test_vector_store as chroma_fixture
from test_bm25 import catalog
from test_context import AS_OF, chunk, family, hit
from test_hybrid import DenseFixture
from northstar_atlas.ingestion.document_loader import APPROVED
from northstar_atlas.vector_store import contracts
from northstar_atlas.vector_store.bm25 import BM25Store
from northstar_atlas.vector_store.hybrid import HybridStore
from northstar_atlas.vector_store.records import VectorStoreError

IT = ('IT Operations',)
HR = ('Human Resources',)


class ScopeTests(unittest.TestCase):
    def require_scope(self, store):
        self.assertIn('department_scope', inspect.signature(store.search).parameters,
                      'pre-ranking department scope has not been implemented')

    def fixture(self, cls):
        cls.setUpClass()
        f = cls()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.require_scope(f.store)
        return f

    def test_scope_validation_rejects_untrusted_shapes_and_normalizes_order(self):
        validate = getattr(contracts, 'validate_department_scope', None)
        self.assertTrue(callable(validate), 'scope validation is missing')
        self.assertIsNone(validate(None))
        self.assertEqual(validate(('IT Operations', 'Human Resources')), ('Human Resources', 'IT Operations'))
        for invalid in ((), [], ['IT Operations'], 'IT Operations', ('unknown',),
                        ('IT Operations', 'IT Operations'), (None,), ({'$ne':'x'},)):
            with self.subTest(invalid=invalid), self.assertRaises(VectorStoreError): validate(invalid)

    def test_bm25_scopes_statistics_before_top_k_without_mutating_catalog(self):
        a = chunk('access', key='it')
        b = replace(chunk('access access access', key='hr', doc='HR-001'), department=HR[0])
        store = BM25Store(catalog([a,b])); self.require_scope(store)
        options = dict(corpus_fingerprint='fixture', mode='draft_review', reference_date=AS_OF, top_k=1)
        selected = store.search('access', department_scope=IT, **options)
        self.assertEqual(selected, BM25Store(catalog([a])).search('access', **options))
        self.assertEqual(selected[0].chunk, a)
        self.assertEqual(store.catalog.chunks, (a,b))
        self.assertEqual(store.search('access', **options), store.search('access', department_scope=None, **options))

    def test_bm25_scope_never_relaxes_authority_and_keeps_siblings(self):
        original = chunk('Intro.\nRequired approval.\nStop.\n')
        parts = family(original, [7])
        store = BM25Store(catalog([original], parts)); self.require_scope(store)
        options = dict(corpus_fingerprint='fixture', reference_date=AS_OF, department_scope=IT)
        self.assertEqual(store.search('approval', **options), ())
        result = store.search('approval', mode='draft_review', top_k=1, **options)
        self.assertEqual({c.chunk_id for c in (result[0].chunk,*result[0].context)}, {c.chunk_id for c in parts})

    def test_chroma_applies_scope_before_top_k_and_preserves_default(self):
        f=self.fixture(chroma_fixture.StoreTests); f.index()
        before=f.search(mode='draft_review')
        result=f.search(mode='draft_review', top_k=1, department_scope=IT)
        self.assertEqual([h.chunk for h in result], [f.it])
        self.assertEqual(f.search(department_scope=IT), ())
        self.assertEqual(f.search(mode='draft_review', department_scope=None), before)
        with self.assertRaises(VectorStoreError): f.search(department_scope=())

    def test_pinecone_filter_keeps_publication_and_authority_constraints(self):
        f=self.fixture(pinecone_fixture.PineconeTests); f.store.synchronize()
        before=f.search(mode='draft_review')
        result=f.search(mode='draft_review', top_k=1, department_scope=IT)
        self.assertEqual([h.chunk for h in result], [f.it])
        path, body, _ = next(v for v in reversed(f.http.calls) if v[0].endswith('/search'))
        self.assertIn(f.store.namespace, path)
        clauses=body['filter']['$and']
        self.assertIn({'department':{'$in':['IT Operations']}}, clauses)
        self.assertTrue(any('snapshot_id' in c for c in clauses))
        self.assertTrue(any('config_digest' in c for c in clauses))
        self.assertTrue(any('status' in c for c in clauses))
        self.assertEqual(f.search(department_scope=IT), ())
        self.assertEqual(f.search(mode='draft_review',department_scope=None), before)

    def test_pinecone_rejects_out_of_scope_response_even_if_eligible(self):
        f=self.fixture(pinecone_fixture.PineconeTests); f.store.synchronize()
        f.http.ignore_filter=True
        with self.assertRaises(VectorStoreError):
            f.search(mode='draft_review',top_k=1,department_scope=IT)

    def test_invalid_scope_is_rejected_before_database_or_embedding_io(self):
        f=self.fixture(pinecone_fixture.PineconeTests)
        for invalid in ((), ('invented',), HR+HR):
            with self.assertRaises(VectorStoreError): f.search(department_scope=invalid)
        self.assertEqual(f.http.calls, [])

    def test_hybrid_propagates_scope_and_rejects_noncompliant_channel(self):
        a=chunk('access',key='it')
        b=replace(chunk('access',key='hr',doc='HR-001'),department=HR[0])
        cat=catalog([a,b]); dense=DenseFixture([hit(a)])
        store=HybridStore(dense,BM25Store(cat)); self.require_scope(store)
        opts=dict(corpus_fingerprint='fixture',mode='draft_review',reference_date=AS_OF)
        result=store.search('access',object(),department_scope=IT,**opts)
        self.assertEqual([h.chunk for h in result], [a])
        self.assertEqual(dense.calls[0][2]['department_scope'], IT)
        self.assertEqual(len(result[0].contributions),2)
        dense.hits=(hit(b),)
        with self.assertRaises(VectorStoreError): store.search('access',object(),department_scope=IT,**opts)


if __name__ == '__main__': unittest.main()
