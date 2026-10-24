"""Score normalization and paired experiment contracts, not model quality tests."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_context import AS_OF, MODEL, ROOT, chunk, family, hit
from test_bm25 import catalog
from test_hybrid import lexical
from northstar_atlas.evaluation.retrieval import EvidenceCatalog, score_question, format_results
from northstar_atlas.evaluation.hybrid import evaluate_case, context_diagnostics, write_results, aks_diagnostic
from northstar_atlas.llm import GenerationLimits
from northstar_atlas.vector_store.bm25 import BM25Store
from northstar_atlas.vector_store.hybrid import HybridStore
from northstar_atlas.vector_store.records import VectorStoreError


def question(refs):
    return {"question_id":"FIXTURE-1","question":"approval", "category":"B", "retrieval_mode":"draft_review",
            "expected_behavior":"answer", "required_evidence":[{"document_id":d,"section_id":s} for d,s in refs]}


class SharedFixture:
    model=MODEL
    def __init__(self): self.calls=[]
    def embed_query(self,text): self.calls.append(text); return (1.,0.,0.)


class DenseFixture:
    def __init__(self,hits): self.hits=hits; self.vectors=[]; self.options=[]
    def search(self,text,provider,**kwargs):
        self.vectors.append(provider.embed_query(text)); self.options.append(kwargs)
        return self.hits[:kwargs['top_k']]


class HybridEvaluationTests(unittest.TestCase):
    def test_dense_and_lexical_use_identical_coverage_scoring(self):
        a=chunk("approval"); b=chunk("other",key="b",section="4.2")
        q=question([(a.document_id,a.section_id),(b.document_id,b.section_id)])
        c=EvidenceCatalog([a,b])
        options=dict(reference_date=AS_OF,latency_seconds=.1)
        dense=score_question(q,[hit(a)],c,**options)
        bm25=score_question(q,[lexical(a)],c,**options)
        self.assertEqual(dense['direct'],bm25['direct'])
        self.assertEqual(dense['expanded'],bm25['expanded'])
        self.assertNotIn('cosine_distance',bm25['retrieved'][0])
        self.assertEqual(bm25['retrieved'][0]['score_kind'],'bm25')
        self.assertEqual(bm25['direct']['missing_evidence'],[{'document_id':b.document_id,'section_id':'4.2'}])

    def test_paired_case_shares_vectors_and_preserves_modes(self):
        a=chunk("approval"); c=catalog([a]); dense=DenseFixture([hit(a)])
        shared=SharedFixture(); q=question([(a.document_id,a.section_id)])
        result=evaluate_case(q,dense,BM25Store(c),shared,reference_date=AS_OF)
        self.assertEqual(set(result['arms']),{'dense','bm25','hybrid'})
        self.assertEqual(dense.vectors,[(1.,0.,0.),(1.,0.,0.)])
        self.assertEqual([v['top_k'] for v in dense.options],[5,20])
        self.assertTrue(all(v['mode']=='draft_review' and v['reference_date']==AS_OF for v in dense.options))
        self.assertTrue(all(r['expanded']['recall_at_5']==1 for r in result['arms'].values()))
        self.assertEqual(result['dense20_prefix']['direct']['recall_at_5'],1)
        self.assertTrue(all(r['prepared_context']['generation_ready'] for r in result['arms'].values()))

    def test_sibling_metrics_remain_separate_with_no_extra_ranks(self):
        original=chunk("Intro.\napproval.\nStop.\n"); parts=family(original,[7])
        c=EvidenceCatalog([original],parts); q=question([(original.document_id,original.section_id)])
        row=score_question(q,[lexical(parts[0],siblings=parts[1:])],c,reference_date=AS_OF,latency_seconds=0)
        self.assertEqual(row['direct']['recall_at_5'],0)
        self.assertEqual(row['expanded']['recall_at_5'],1)
        self.assertIsNone(row['sibling_context'][0]['rank'])
        self.assertNotIn('score',row['sibling_context'][0])

    def test_discarded_candidates_retain_ranks_typed_scores_and_sibling_ids(self):
        chunks=[chunk('approval',key=str(i),section=f'4.{i}') for i in range(1,7)]
        c=catalog(chunks)
        result=evaluate_case(question([(chunks[0].document_id,chunks[0].section_id)]),
                             DenseFixture([hit(v) for v in chunks]),BM25Store(c),SharedFixture(),reference_date=AS_OF)
        candidates=result['stages']['candidate_records']
        self.assertEqual(len(result['arms']['hybrid']['retrieved']),5)
        self.assertEqual([c['rank'] for c in candidates['dense']],list(range(1,7)))
        self.assertEqual(candidates['dense'][-1]['cosine_distance'],.2)
        self.assertEqual(candidates['bm25'][-1]['score_kind'],'bm25')
        self.assertGreater(candidates['bm25'][-1]['score'],0)
        self.assertEqual(candidates['dense'][-1]['required_context_ids'],[])
        self.assertEqual(candidates['dense'][-1]['content_hash'],chunks[-1].content_hash)

    def test_budget_failure_is_not_reported_as_delivered_evidence(self):
        a=chunk("Approval condition.\n"*100)
        q=question([(a.document_id,a.section_id)])
        info=context_diagnostics(q,[lexical(a)],EvidenceCatalog([a]),reference_date=AS_OF,
                                 limits=GenerationLimits(context_tokens=2048,output_tokens=1024,margin_tokens=512))
        self.assertFalse(info['generation_ready'])
        self.assertEqual(info['delivered_evidence']['recall'],0)
        self.assertTrue(info['warnings'])

    def test_bad_channel_stops_evaluation_instead_of_scoring_rescued_text(self):
        a=chunk("approval"); c=catalog([a]); q=question([(a.document_id,a.section_id)])
        with self.assertRaises(VectorStoreError):
            evaluate_case(q,DenseFixture([hit(replace(a,text="forged"))]),BM25Store(c),SharedFixture(),reference_date=AS_OF)

    def test_aks_is_exact_separate_and_has_provisioning_references(self):
        q=aks_diagnostic()
        self.assertEqual(q['question'],"An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?")
        self.assertEqual(q['retrieval_mode'],'draft_review')
        self.assertIn({'document_id':'IT-001','section_id':'9.1'},q['required_evidence'])
        self.assertIn({'document_id':'SEC-001','section_id':'6.1'},q['required_evidence'])
        frozen=json.loads((ROOT/'evaluations/questions.json').read_text())
        self.assertNotIn(q['question_id'],[r['question_id'] for r in frozen['questions']])

    def test_deterministic_formatting_exclusive_local_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.local/tmp') as temp:
            root=Path(temp)
            p=write_results(root,{'b':2,'a':1},run_name='fixture')
            self.assertEqual(p,root/'.local/evaluations/m5/fixture.json')
            self.assertEqual(p.read_text(),format_results({'a':1,'b':2}))
            with self.assertRaises(FileExistsError): write_results(root,{},run_name='fixture')
            with self.assertRaises(ValueError): write_results(root,{},run_name='../escape')


if __name__ == '__main__': unittest.main()
