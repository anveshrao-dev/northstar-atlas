"""Small deterministic BM25 variant; no embedding model, database or SDK imports."""
from collections import Counter
import math
import re
import unicodedata

from .catalog import ChunkCatalog, eligible, search_options
from .contracts import RankedHit, validate_department_scope
from .records import VectorStoreError

TOKENIZER_VERSION = "unicode-word-v1"
ALGORITHM_VERSION = "bm25-positive-idf-v1"


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


class BM25Store:
    def __init__(self, catalog: ChunkCatalog, *, k1: float = 1.2, b: float = 0.75):
        if (type(k1) not in (int, float) or not math.isfinite(k1) or k1 <= 0
                or type(b) not in (int, float) or not math.isfinite(b) or not 0 <= b <= 1):
            raise VectorStoreError("invalid BM25 k1/b parameters")
        self.catalog, self.k1, self.b = catalog, k1, b
        self.terms = {c.chunk_id: Counter(tokenize(c.text)) for c in catalog.chunks}
        self.lengths = {key: sum(terms.values()) for key, terms in self.terms.items()}
        if not sum(self.lengths.values()):
            raise VectorStoreError("catalog contains no lexical tokens")

    def search(self, query, provider=None, *, corpus_fingerprint, mode="approved_only", top_k=5,
               reference_date=None, department_scope=None) -> tuple[RankedHit, ...]:
        scope = validate_department_scope(department_scope)
        as_of = search_options(query, mode, top_k, reference_date)
        self.catalog.check_current(corpus_fingerprint)
        chunks = tuple(c for c in self.catalog.chunks if eligible(c, mode, as_of)
                       and (scope is None or c.department in scope))
        query_terms = sorted(set(tokenize(query)))
        if not chunks or not query_terms:
            return ()
        average = sum(self.lengths[c.chunk_id] for c in chunks) / len(chunks)
        if average == 0:
            raise VectorStoreError("eligible catalog contains no lexical tokens")
        df = Counter(t for c in chunks for t in self.terms[c.chunk_id])
        idf = {t: math.log1p((len(chunks) - df[t] + .5) / (df[t] + .5)) for t in query_terms}
        hits = []
        for c in chunks:
            terms = self.terms[c.chunk_id]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[c.chunk_id] / average)
            score = sum(idf[t] * terms[t] * (self.k1 + 1) / (terms[t] + norm)
                        for t in query_terms if terms[t])
            if score > 0:
                hits.append(RankedHit(c, score, "bm25", self.catalog.config.model,
                                      self.catalog.siblings(c, mode, as_of)))
        self.catalog.check_current(corpus_fingerprint)
        return tuple(sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k])
