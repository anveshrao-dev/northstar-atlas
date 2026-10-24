"""Read-only evaluation sessions using prior publication evidence, not full scans.

These opt-in subclasses leave ordinary production search/audit defaults intact.
They detect stale receipts/counts and validate returned payloads, but cannot prove
that an unqueried remote record has not changed since the prior full audit.
"""

from dataclasses import asdict
from contextlib import contextmanager
import math

from northstar_atlas.embeddings import validate_vector
from northstar_atlas.vector_store.pinecone_cloud import as_dict
from northstar_atlas.vector_store.pinecone_store import PineconeStore
from northstar_atlas.vector_store.records import VectorStoreError, digest, metadata_for
from northstar_atlas.vector_store.store import ChromaStore


class ComparisonChromaStore(ChromaStore):
    def __init__(self, root, snapshot, receipt):
        super().__init__(root, snapshot.config)
        self.snapshot = snapshot
        self.records = {r.chunk_id:r for r in snapshot.records}
        self.chunks = {r.chunk_id:r.chunk for r in snapshot.records}
        self.receipt = {k:receipt.get(k) for k in
                        ('state','count','config','corpus_fingerprint','record_digest','vector_digest')}
        expected = {r.chunk_id:(r.chunk.text,metadata_for(r.chunk,r.model)) for r in snapshot.records}
        if (self.receipt['state'] != 'ready' or self.receipt['count'] != len(expected)
                or self.receipt['config'] != self.config.serialized
                or self.receipt['corpus_fingerprint'] != snapshot.corpus_fingerprint
                or self.receipt['record_digest'] != digest(expected)
                or not isinstance(self.receipt['vector_digest'],str)):
            raise VectorStoreError('Chroma publication receipt does not match comparison snapshot')
        self.checked_ids = set()
        self.max_direction_error = 0.0

    def _ready(self, collection, state):
        if any(state.get(k) != v for k,v in self.receipt.items()) or collection.count() != len(self.chunks):
            raise VectorStoreError('Chroma publication/count changed; stop comparison and investigate')
        return self.chunks

    def light_status(self):
        with self._lock():
            collection = self._collection()
            self._ready(collection,self._read_state())
            return {**self.receipt,'verification':'prior publication plus current config/count; no full fetch'}

    def search(self,*args,**kwargs):
        hits = super().search(*args,**kwargs)
        ids = sorted({c.chunk_id for h in hits for c in (h.chunk,*h.context)})
        if not ids: return hits
        # The inherited ranking and sibling rules use known chunks. Validate
        # their actual stored payloads before returning any of that context.
        with self._lock():
            collection = self._collection()
            self._ready(collection,self._read_state())
            result = collection.get(ids=ids,include=['documents','metadatas','embeddings'])
            if sorted(result['ids']) != ids: raise VectorStoreError('missing selected Chroma record/sibling')
            for key,text,meta,vector in zip(result['ids'],result['documents'],result['metadatas'],result['embeddings'],strict=True):
                expected = self.records[key]
                if text != expected.chunk.text or meta != metadata_for(expected.chunk,expected.model):
                    raise VectorStoreError('selected Chroma record differs from verified snapshot')
                values = validate_vector(vector.tolist(),dimension=self.config.model.dimension)
                norm, expected_norm = math.hypot(*values), math.hypot(*expected.vector)
                error = max(abs(a/norm-b/expected_norm) for a,b in zip(values,expected.vector,strict=True))
                if error > 1e-6: raise VectorStoreError('selected Chroma vector direction differs from frozen snapshot')
                self.max_direction_error = max(self.max_direction_error,error)
                self.checked_ids.add(key)
        return hits


class ComparisonPineconeStore(PineconeStore):
    def __init__(self,*args,receipt,describe_namespace,**kwargs):
        super().__init__(*args,**kwargs)
        self.describe_namespace = describe_namespace
        if (receipt.get('state') != 'ready' or receipt.get('remote_audited') is not True
                or receipt.get('namespace') != self.namespace or receipt.get('count') != len(self.by_id)
                or receipt.get('target') != asdict(self.target)
                or receipt.get('embedding_config') != asdict(self.config)
                or receipt.get('corpus_fingerprint') != self.snapshot.corpus_fingerprint):
            raise VectorStoreError('Pinecone prior audit receipt does not match snapshot/target')
        self.receipt = receipt
        self.initial_run = self.state.read_run(self.namespace)

    def light_status(self):
        with self.state.lock():
            self._ready()
            self._target_check()
            if self.state.read_run(self.namespace) != self.initial_run:
                raise VectorStoreError('Pinecone publication changed during comparison')
            info = as_dict(self._call('describe namespace',self.describe_namespace))
            count = info.get('record_count')
            if (info.get('name') != self.namespace or isinstance(count,bool)
                    or str(count) != str(len(self.by_id))):
                raise VectorStoreError('Pinecone namespace count/identity mismatch; stop comparison')
            self._audited = True
            return {**self.status(),'count':int(count),'verification':'prior full audit plus current target/publication/count; no full fetch'}

    def _audit_or_invalidate(self):
        # This method is called under the inherited search lock. Require an
        # explicit light_status at session startup rather than nesting locks.
        raise VectorStoreError('comparison session requires lightweight readiness before search')

    @contextmanager
    def _search_integrity(self):
        # Production search invalidates its publication after corrupt responses.
        # Evaluation must stop without changing that production state.
        try:
            yield
        except VectorStoreError:
            self._audited = False
            raise

    def audit(self):
        raise VectorStoreError('full audits are not part of this evaluation session')

    def synchronize(self):
        raise VectorStoreError('comparison sessions cannot upload')
