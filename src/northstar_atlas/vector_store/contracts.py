"""Small shared retrieval contract; no backend SDK or network initialization."""

from dataclasses import dataclass
from datetime import date
import math
from typing import Literal, Protocol

from northstar_atlas.embeddings import EmbeddingModel, EmbeddingProvider
from northstar_atlas.ingestion.chunker import PolicyChunk
from northstar_atlas.ingestion.document_loader import RetrievalMode
from .records import VectorStoreError


@dataclass(frozen=True)
class SearchHit:
    chunk: PolicyChunk
    cosine_distance: float
    model: EmbeddingModel
    context: tuple[PolicyChunk, ...]

    @property
    def cosine_similarity(self) -> float:
        """1 - cosine distance; neither quantity is answer confidence."""
        return 1.0 - self.cosine_distance


@dataclass(frozen=True)
class ScoreContribution:
    channel: Literal["dense", "bm25"]
    rank: int
    score: float
    score_kind: Literal["cosine_distance", "bm25"]


@dataclass(frozen=True)
class RankedHit:
    """Non-dense rank evidence; model identifies chunk provenance, not BM25 encoding."""
    chunk: PolicyChunk
    score: float
    score_kind: Literal["bm25", "rrf"]
    model: EmbeddingModel
    context: tuple[PolicyChunk, ...]
    contributions: tuple[ScoreContribution, ...] = ()


RetrievalHit = SearchHit | RankedHit


def score_fields(hit: RetrievalHit) -> dict:
    """Validate scores and serialize without changing the legacy dense output."""
    if isinstance(hit, SearchHit):
        if type(hit.cosine_distance) not in (int, float) or not math.isfinite(hit.cosine_distance):
            raise VectorStoreError("nonfinite or invalid retrieval distance")
        return {"cosine_distance": hit.cosine_distance, "cosine_similarity": hit.cosine_similarity}
    if (not isinstance(hit, RankedHit) or hit.score_kind not in ("bm25", "rrf")
            or type(hit.score) not in (int, float) or not math.isfinite(hit.score) or hit.score <= 0):
        raise VectorStoreError("invalid ranked search score")
    contributions = []
    seen = set()
    for c in hit.contributions:
        if (not isinstance(c, ScoreContribution) or c.channel not in ("dense", "bm25")
                or c.channel in seen or type(c.rank) is not int or c.rank < 1
                or c.score_kind != {"dense": "cosine_distance", "bm25": "bm25"}[c.channel]
                or type(c.score) not in (int, float) or not math.isfinite(c.score)
                or (c.channel == "bm25" and c.score <= 0)):
            raise VectorStoreError("invalid rank contribution")
        seen.add(c.channel)
        contributions.append({"channel": c.channel, "rank": c.rank,
                              "score": c.score, "score_kind": c.score_kind})
    return {"score_kind": hit.score_kind, "score": hit.score,
            "higher_is_better": True, "contributions": contributions}


class RetrievalStore(Protocol):
    def search(self, query: str, provider: EmbeddingProvider | None, *, corpus_fingerprint: str,
               mode: RetrievalMode = "approved_only", top_k: int = 5,
               reference_date: date | None = None) -> tuple[RetrievalHit, ...]: ...
