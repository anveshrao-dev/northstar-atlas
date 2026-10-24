"""Small shared retrieval contract; no backend SDK or network initialization."""

from dataclasses import dataclass
from datetime import date
from typing import Protocol

from northstar_atlas.embeddings import EmbeddingModel, EmbeddingProvider
from northstar_atlas.ingestion.chunker import PolicyChunk
from northstar_atlas.ingestion.document_loader import RetrievalMode


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


class RetrievalStore(Protocol):
    def search(self, query: str, provider: EmbeddingProvider, *, corpus_fingerprint: str,
               mode: RetrievalMode = "approved_only", top_k: int = 5,
               reference_date: date | None = None) -> tuple[SearchHit, ...]: ...
