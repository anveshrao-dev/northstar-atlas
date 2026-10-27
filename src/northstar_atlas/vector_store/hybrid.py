"""Bounded rank fusion; siblings carry evidence but never extra rank votes."""
from datetime import date

from .bm25 import BM25Store
from .catalog import ChunkCatalog, search_options
from .contracts import RankedHit, RetrievalStore, ScoreContribution, SearchHit, validate_department_scope
from .records import VectorStoreError

CANDIDATES = 20
RRF_CONSTANT = 60


def fuse_results(dense, lexical, catalog: ChunkCatalog, *, mode: str, reference_date: date,
               top_k: int = 5, department_scope=None) -> tuple[RankedHit, ...]:
    scope = validate_department_scope(department_scope)
    search_options("fusion", mode, top_k, reference_date)
    if top_k > 5:
        raise VectorStoreError("hybrid final top_k must be 1..5")
    catalog.check_current(catalog.fingerprint)
    pool, contributions = {}, {}
    for channel, values in (("dense", tuple(dense)), ("bm25", tuple(lexical))):
        if len(values) > CANDIDATES:
            raise VectorStoreError("channel exceeded candidate limit")
        seen = set()
        for rank, hit in enumerate(values, 1):
            if ((channel == "dense" and not isinstance(hit, SearchHit)) or
                    (channel == "bm25" and (not isinstance(hit, RankedHit) or hit.score_kind != "bm25"))):
                raise VectorStoreError("channel returned an incompatible score type")
            catalog.validate_hit(hit, mode, reference_date)
            if scope is not None and hit.chunk.department not in scope:
                raise VectorStoreError('channel returned out-of-scope evidence')
            key = hit.chunk.chunk_id
            if key in seen:
                raise VectorStoreError("duplicate direct chunk ID within a channel")
            seen.add(key)
            pool[key] = hit.chunk
            contributions.setdefault(key, []).append(ScoreContribution(
                channel, rank, hit.cosine_distance if channel == "dense" else hit.score,
                "cosine_distance" if channel == "dense" else "bm25"))
    hits = [RankedHit(chunk, sum(1 / (RRF_CONSTANT + c.rank) for c in contributions[key]), "rrf",
                      catalog.config.model, catalog.siblings(chunk, mode, reference_date), tuple(contributions[key]))
            for key, chunk in pool.items()]
    return tuple(sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k])


class HybridStore:
    def __init__(self, dense: RetrievalStore, lexical: BM25Store):
        self.dense, self.lexical = dense, lexical

    def search(self, query, provider, *, corpus_fingerprint, mode="approved_only", top_k=5,
               reference_date=None, department_scope=None) -> tuple[RankedHit, ...]:
        scope = validate_department_scope(department_scope)
        as_of = search_options(query, mode, top_k, reference_date)
        if provider is None or top_k > 5:
            raise VectorStoreError("hybrid requires a dense embedder and final top_k 1..5")
        self.lexical.catalog.check_current(corpus_fingerprint)
        options = dict(corpus_fingerprint=corpus_fingerprint, mode=mode, top_k=CANDIDATES, reference_date=as_of)
        if scope is not None:
            options['department_scope'] = scope
        dense = self.dense.search(query, provider, **options)
        lexical = self.lexical.search(query, None, **options)
        return fuse_results(dense, lexical, self.lexical.catalog, mode=mode, reference_date=as_of,
                            top_k=top_k, department_scope=scope)
