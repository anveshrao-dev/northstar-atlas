"""Paired read-only Chroma/Pinecone evaluation with shared query vectors."""

from northstar_atlas.project import ProjectRootError, validate_project_root

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter

from northstar_atlas.embeddings import embed_query
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.local_embeddings import MiniLMProvider
from northstar_atlas.vector_store.pinecone_cloud import (
    CloudError, check_confirmation, load_settings, local_path, sdk_client,
)
from northstar_atlas.vector_store.pinecone_workflow import RequestMeter, active_snapshot, connect_target
from northstar_atlas.vector_store.preparation import corpus_snapshot, float32_vector
from northstar_atlas.vector_store.records import VectorStoreError, digest
from northstar_atlas.vector_store.store import _vector_digest
from .backends import ComparisonChromaStore, ComparisonPineconeStore
from .retrieval import EvidenceCatalog, K, SCORING_VERSION, format_results, load_dataset, score_question, summarize


class SharedQueries:
    """The model encodes each exact question once; backends receive identical float32 values."""
    def __init__(self,questions,provider,config):
        if provider.model != config.model: raise ValueError('query model does not match index model')
        self.model, self.max_input_tokens, self.config = provider.model,config.token_limit,config
        self.vectors,self.timings = {},{}
        for q in questions:
            text = q['question']
            if text in self.vectors: continue
            started = perf_counter()
            self.vectors[text] = float32_vector(embed_query(text,provider).vector,config)
            self.timings[text] = perf_counter()-started

    def embed_query(self,text):
        if text not in self.vectors: raise ValueError('query was not in the prepared comparison set')
        return self.vectors[text]

    def embed_documents(self,texts):
        raise ValueError('document encoding is forbidden during comparison')


def normalize_hits(hits,config):
    """Both adapters already return distances; preserve ranks and reject incompatible data."""
    hits = tuple(hits)
    if len(hits)>K or len({h.chunk.chunk_id for h in hits}) != len(hits):
        raise ValueError('invalid ranked hit count/duplicates')
    if any(h.model != config.model or not math.isfinite(h.cosine_distance)
           or not -1e-6 <= h.cosine_distance <= 2.000001 for h in hits):
        raise ValueError('invalid backend model or cosine distance')
    return hits


def compare_rankings(chroma,pinecone):
    a={h.chunk.chunk_id:(i,h) for i,h in enumerate(chroma,1)}
    b={h.chunk.chunk_id:(i,h) for i,h in enumerate(pinecone,1)}
    return {'chroma_ids':list(a),'pinecone_ids':list(b),'identical_order':list(a)==list(b),
            'identical_set':set(a)==set(b),'overlap_count':len(set(a)&set(b)),
            'only_chroma':sorted(set(a)-set(b)),'only_pinecone':sorted(set(b)-set(a)),
            'common':[{'chunk_id':key,'chroma_rank':a[key][0],'pinecone_rank':b[key][0],
                       'pinecone_minus_chroma_rank':b[key][0]-a[key][0],
                       'similarity_delta':b[key][1].cosine_similarity-a[key][1].cosine_similarity}
                      for key in a if key in b]}


def evaluate_questions(questions,backends,shared,catalog,*,fingerprint,reference_date):
    if set(backends) != {'chroma','pinecone'}: raise ValueError('both comparison backends required')
    rows={'chroma':[],'pinecone':[],'rankings':[]}
    for i,q in enumerate(questions):
        hits={}
        order=('chroma','pinecone') if i%2==0 else ('pinecone','chroma')
        for name in order:
            started=perf_counter()
            hits[name]=normalize_hits(backends[name].search(q['question'],shared,
                corpus_fingerprint=fingerprint,mode=q['retrieval_mode'],top_k=K,reference_date=reference_date),shared.config)
            elapsed=perf_counter()-started
            row=score_question(q,hits[name],catalog,reference_date=reference_date,latency_seconds=elapsed)
            if row['violations']: raise ValueError('retrieval eligibility/provenance/context violation; comparison stopped')
            row.update(query_embedding_seconds=shared.timings[q['question']],
                       query_vector_digest=digest(shared.embed_query(q['question'])),backend_order=list(order))
            rows[name].append(row)
        rows['rankings'].append({'question_id':q['question_id'],**compare_rankings(hits['chroma'],hits['pinecone'])})
    return rows


def latency_summary(values):
    values=list(values)
    return {'mean_seconds':mean(values),'median_seconds':median(values),'max_seconds':max(values),
            'first_seconds':values[0],'remaining_mean_seconds':mean(values[1:]) if len(values)>1 else None}


def _receipt(root,pattern,predicate):
    paths=sorted(local_path(root,'.local/pinecone/m33').glob(pattern),reverse=True)
    for path in paths:
        path=local_path(root,path.relative_to(root).as_posix())
        value=json.loads(path.read_text())
        if predicate(value): return value,path.relative_to(root).as_posix()
    raise ValueError('matching prior M3.3 verification receipt missing')


def run_comparison(root,reference_date):
    started=perf_counter()
    dataset,dataset_hash=load_dataset(root)
    documents,fingerprint=corpus_snapshot(root)
    settings=load_settings(root);check_confirmation(root,settings)
    meter=RequestMeter()
    with sdk_client(settings) as client:
        meter.attach(client)
        target,index=connect_target(client,meter,settings)
        with index:
            snapshot=active_snapshot(root,target)
            receipt,receipt_path=_receipt(root,'*-status.json',lambda d:d.get('remote_audited') is True and d.get('namespace')==snapshot.namespace)
            chroma_receipt_path=local_path(root,'.local/m33/chroma-status.json')
            chroma_receipt=json.loads(chroma_receipt_path.read_text())
            chroma=ComparisonChromaStore(root,snapshot,chroma_receipt)
            pinecone=ComparisonPineconeStore(root,snapshot,target,receipt=receipt,
                documents=index.documents,describe_target=lambda:client.indexes.describe(name=target.name),
                describe_namespace=lambda:index.describe_namespace(name=snapshot.namespace),
                current_fingerprint=lambda:corpus_snapshot(root)[1])
            before={'chroma':chroma.light_status(),'pinecone':pinecone.light_status()}
            status_usage=meter.report()
            provider=MiniLMProvider(root)
            canonical=tuple(c for d in documents for c in chunk_document(d).chunks)
            catalog=EvidenceCatalog(canonical,[r.chunk for r in snapshot.records])
            shared=SharedQueries(dataset['questions'],provider,snapshot.config)
            setup_seconds=perf_counter()-started
            prior_usage=meter.report()
            rows=evaluate_questions(dataset['questions'],{'chroma':chroma,'pinecone':pinecone},shared,catalog,
                                    fingerprint=fingerprint,reference_date=reference_date)
            query_usage=meter.report()
            after={'chroma':chroma.light_status(),'pinecone':pinecone.light_status()}
            if before!=after or corpus_snapshot(root)[1]!=fingerprint or load_dataset(root)[1]!=dataset_hash:
                raise ValueError('index/corpus/dataset changed; comparison invalid')
            counts=Counter(query_usage['logical_http_calls']);counts.subtract(prior_usage['logical_http_calls'])
            result={'schema_version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                'reference_date':reference_date.isoformat(),'top_k':K,'scoring_version':SCORING_VERSION,
                'scoring':'M2.5 complete source-section coverage; parents require own text plus descendants; expanded RR uses direct trigger prefix',
                'semantic_support_adjudicated':False,'answer_faithfulness_evaluated':False,
                'provenance':{'dataset_sha256':dataset_hash,'manifest_sha256':dataset['corpus_manifest_sha256'],
                    'corpus_fingerprint':fingerprint,'namespace':snapshot.namespace,'config':asdict(snapshot.config),
                    'sources':[{'document_id':d.document_id,'content_hash':d.content_hash,'status':d.status} for d in documents],
                    'pinecone_audit_receipt':receipt_path,'chroma_audit_receipt':str(chroma_receipt_path.relative_to(root)),
                    'before':before,'after':after,'query_vectors':shared.vectors,
                    'query_matrix_digest':digest(shared.vectors),
                    'document_vector_byte_digest_equal':_vector_digest({r.chunk_id:r.vector for r in snapshot.records})==chroma_receipt['vector_digest'],
                    'chroma_selected_vectors_checked':len(chroma.checked_ids),'max_selected_direction_error':chroma.max_direction_error,
                    'packages':{n:importlib.metadata.version(n) for n in ('chromadb','pinecone','sentence-transformers','torch','transformers','numpy')}},
                'timing':{'setup_seconds':setup_seconds,'model_load_seconds':provider.load_seconds,
                    'query_encoding':latency_summary(shared.timings[q['question']] for q in dataset['questions']),
                    **{name:latency_summary(r['latency_seconds'] for r in rows[name]) for name in ('chroma','pinecone')},
                    'definition':'retrieval includes readiness/source/target checks, search and selected payload/sibling checks; query model encoding, setup, scoring and artifact IO excluded',
                    'ordering':'one pass in dataset order; alternate backend order; no query warmup'},
                'usage':{'initial_status':status_usage,'all_operations':meter.report(),
                    'query_phase':{'logical_http_calls':dict(sorted((k,v) for k,v in counts.items() if v)),
                                   'reported_read_units':query_usage['reported_read_units']-prior_usage['reported_read_units']}},
                'summary':{name:summarize(rows[name]) for name in ('chroma','pinecone')},'results':rows}
            return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-date',type=date.fromisoformat,default=date(2026,10,22))
    args=parser.parse_args()
    try:
        root=validate_project_root(Path.cwd())
        result=run_comparison(root,args.reference_date)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        path=local_path(root,f'.local/evaluations/m3.4/comparison-{stamp}.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:stream.write(format_results(result))
        print(format_results({'artifact':str(path.relative_to(root)),'summary':result['summary'],'timing':result['timing'],'usage':result['usage']}))
        return 0
    except (CloudError,VectorStoreError) as exc:
        print(f'Comparison stopped: {exc}',file=sys.stderr);return 1
    except Exception:
        print('Comparison stopped: validation or provider failure; no provider details logged; no rebuild attempted.',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
