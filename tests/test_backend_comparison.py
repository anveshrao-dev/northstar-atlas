"""M3.4 paired evaluation: controlled vectors, scoring and lightweight reads."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_vector_store import ROOT, AS_OF, FakeProvider
from test_pinecone_store import MemoryHTTP
from northstar_atlas.embeddings import embed_chunks
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents
from northstar_atlas.ingestion.refiner import refine_chunks
from northstar_atlas.vector_store.contracts import SearchHit
from northstar_atlas.vector_store.preparation import freeze_snapshot
from northstar_atlas.vector_store.records import IndexConfig, VectorStoreError
from northstar_atlas.vector_store.store import ChromaStore
from northstar_atlas.vector_store.pinecone_store import PineconeStore, PineconeTarget
from northstar_atlas.evaluation.retrieval import EvidenceCatalog, format_results, score_question
from northstar_atlas.evaluation.comparison import SharedQueries, normalize_hits, compare_rankings, evaluate_questions
from northstar_atlas.evaluation.backends import ComparisonChromaStore, ComparisonPineconeStore


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = load_documents(ROOT, mode="draft_review", reference_date=AS_OF)
        cls.hr = chunk_document(docs[0]).chunks[0]
        cls.it = chunk_document(docs[1]).chunks[0]

    def setUp(self):
        self.provider = FakeProvider()
        self.config = IndexConfig(self.provider.model, 256)
        self.snapshot = freeze_snapshot(embed_chunks([self.hr, self.it], self.provider), self.config, "test-corpus")
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="northstar-atlas"\n')
        self.q = {"question_id":"EVAL-001", "question":"Access?", "category":"A", "retrieval_mode":"approved_only",
                  "expected_behavior":"answer", "required_evidence":[{"document_id":self.hr.document_id,"section_id":self.hr.section_id}]}

    def hit(self, chunk, distance=0.2, context=()):
        return SearchHit(chunk, distance, self.provider.model, context)

    def test_query_encoded_once_and_identical_for_both_backends(self):
        with patch.object(self.provider, "embed_query", wraps=self.provider.embed_query) as encode:
            shared = SharedQueries([self.q], self.provider, self.config)
        self.assertEqual(encode.call_count, 1)
        catalog = EvidenceCatalog([self.hr, self.it])
        vectors = []
        class Backend:
            def search(inner, query, provider, **kwargs):
                vectors.append(provider.embed_query(query))
                return (self.hit(self.hr),)
        results = evaluate_questions([self.q], {"chroma":Backend(),"pinecone":Backend()}, shared, catalog,
                                     fingerprint="test-corpus", reference_date=AS_OF)
        self.assertEqual(vectors[0], vectors[1])
        for view in ("direct", "expanded"):
            self.assertEqual(results["chroma"][0][view],results["pinecone"][0][view])
        with self.assertRaises(ValueError): shared.embed_documents(["not allowed"])
        with self.assertRaises(ValueError): shared.embed_query("different question")

    def test_normalization_preserves_ranks_scores_and_rejects_invalid_hits(self):
        hits=(self.hit(self.hr,0.1),self.hit(self.it,0.3))
        self.assertEqual(normalize_hits(hits,self.config),hits)
        self.assertAlmostEqual(hits[0].cosine_similarity,0.9)
        for bad in ((hits[0],hits[0]),(self.hit(self.hr,float('nan')),),
                    (replace(hits[0],model=replace(self.provider.model,revision="wrong")),)):
            with self.assertRaises(ValueError): normalize_hits(bad,self.config)

    def test_rank_comparison_and_deterministic_formatting(self):
        a=[self.hit(self.hr,0.1),self.hit(self.it,0.2)]
        b=[self.hit(self.it,0.09),self.hit(self.hr,0.21)]
        result=compare_rankings(a,b)
        self.assertFalse(result['identical_order'])
        self.assertTrue(result['identical_set'])
        self.assertEqual(result['overlap_count'],2)
        self.assertEqual([x['pinecone_minus_chroma_rank'] for x in result['common']], [1,-1])
        self.assertEqual(format_results(result),format_results(dict(reversed(list(result.items())))))

    def test_missing_evidence_and_status_scoring_use_existing_rules(self):
        catalog=EvidenceCatalog([self.hr,self.it])
        missing=score_question(self.q,[],catalog,reference_date=AS_OF,latency_seconds=0)
        self.assertEqual(missing['direct']['missing_evidence'],self.q['required_evidence'])
        leaked=score_question(self.q,[self.hit(self.it)],catalog,reference_date=AS_OF,latency_seconds=0)
        self.assertTrue(leaked['violations'])
        draft={**self.q,'category':'B','retrieval_mode':'draft_review','required_evidence':[{'document_id':self.it.document_id,'section_id':self.it.section_id}]}
        scored=score_question(draft,[self.hit(self.it)],catalog,reference_date=AS_OF,latency_seconds=0)
        self.assertEqual(scored['direct']['recall_at_5'],1)
        self.assertEqual(scored['retrieved'][0]['status'],self.it.status)

    def test_siblings_are_unranked_and_only_expand_coverage(self):
        text="## 1. Rule\n\n" + "\n\n".join("An authorized owner reviews this access request and records its conditions." for _ in range(30))
        canonical=replace(self.hr,text=text,source_start=0,source_end=len(text))
        refined=refine_chunks([canonical],self.provider).require_complete()
        self.assertGreater(len(refined),1)
        result=score_question(self.q,[self.hit(refined[0],context=tuple(refined[1:]))],
                              EvidenceCatalog([canonical],refined),reference_date=AS_OF,latency_seconds=0)
        self.assertEqual(result['direct']['recall_at_5'],0)
        self.assertEqual(result['expanded']['recall_at_5'],1)
        self.assertTrue(all(c['rank'] is None for c in result['sibling_context']))

    def test_chroma_light_checks_never_fetch_entire_collection(self):
        original=ChromaStore(self.root,self.config)
        original.synchronize(self.snapshot.records,corpus_fingerprint='test-corpus')
        receipt=original.status()
        store=ComparisonChromaStore(self.root,self.snapshot,receipt)
        with patch.object(store,'_records',side_effect=AssertionError('full scan')), patch.object(store,'_stored_vectors',side_effect=AssertionError('full vectors')):
            self.assertEqual(store.light_status()['count'],2)
            approved=store.search('access',self.provider,corpus_fingerprint='test-corpus',reference_date=AS_OF)
            self.assertEqual([h.chunk.document_id for h in approved],['HR-001'])
            self.assertEqual(len(store.search('access',self.provider,corpus_fingerprint='test-corpus',mode='draft_review')),2)
        original._collection().delete(ids=[self.it.chunk_id])
        with self.assertRaises(VectorStoreError):store.light_status()

    def test_pinecone_light_check_uses_publication_and_count_not_full_audit(self):
        from pinecone.client.documents import Documents
        http=MemoryHTTP(); target=PineconeTarget('northstar-atlas-dev','https://offline.invalid','test-project')
        description={'name':target.name,'host':target.host,'status':{'ready':True},'deletion_protection':'enabled',
                     'deployment':{'deployment_type':'managed','cloud':'azure','region':'eastus2'},
                     'schema':{'fields':{'embedding':{'type':'dense_vector','dimension':3,'metric':'cosine'}}}}
        args=dict(documents=Documents(http=http,host=target.host),describe_target=lambda:description,
                  current_fingerprint=lambda:'test-corpus',poll_attempts=2,poll_interval=0)
        original=PineconeStore(self.root,self.snapshot,target,**args);original.synchronize()
        receipt={**original.status(),'remote_audited':True,'target':asdict(target),'embedding_config':asdict(self.config),'corpus_fingerprint':'test-corpus'}
        count={'name':self.snapshot.namespace,'record_count':'2'}
        store=ComparisonPineconeStore(self.root,self.snapshot,target,receipt=receipt,describe_namespace=lambda:count,**args)
        http.calls.clear()
        with patch.object(store,'_audit',side_effect=AssertionError('full audit')):
            self.assertEqual(store.light_status()['count'],2)
            store.search('access',self.provider,corpus_fingerprint='test-corpus',reference_date=AS_OF)
        self.assertFalse(any(path.endswith('/list') for path,_,_ in http.calls))
        initial=store.state.read_run(store.namespace)
        http.score=2.0
        with patch.object(store.state,'write_run',wraps=store.state.write_run) as write:
            with self.assertRaises(VectorStoreError):
                store.search('access',self.provider,corpus_fingerprint='test-corpus',reference_date=AS_OF)
            write.assert_not_called()
        self.assertEqual(store.state.read_run(store.namespace),initial)
        self.assertFalse(store._audited)
        count['record_count']='1'
        with self.assertRaises(VectorStoreError):store.light_status()


if __name__=='__main__':unittest.main()
