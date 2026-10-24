"""Verified ingestion/indexing and query workflows; no answer generation."""

from dataclasses import asdict
from datetime import date
from pathlib import Path
from time import perf_counter
from northstar_atlas.ingestion.document_loader import RetrievalMode
from .records import IndexConfig
from .preparation import RefiningProvider, corpus_snapshot, prepare_corpus
from .store import ChromaStore, SearchHit


def index_corpus(root: Path, provider: RefiningProvider) -> dict:
    """Provider supplies both the existing embedding protocol and tokenizer counter."""
    started = perf_counter()
    prepared = prepare_corpus(root, provider)
    report = ChromaStore(root, prepared.config).synchronize(prepared.records, corpus_fingerprint=prepared.fingerprint)
    return {**asdict(report), "documents": prepared.documents, "canonical_chunks": prepared.canonical_chunks,
            "embedding_chunks": len(prepared.records), "max_tokens": prepared.max_tokens,
            "warnings": list(prepared.warnings), "embedding_seconds": round(prepared.embedding_seconds, 3),
            "total_seconds": round(perf_counter() - started, 3)}


def search_corpus(root: Path, query: str, provider: RefiningProvider, *,
                  mode: RetrievalMode = "approved_only", top_k: int = 5,
                  reference_date: date | None = None) -> tuple[SearchHit, ...]:
    _, fingerprint = corpus_snapshot(root)
    config = IndexConfig(provider.model, provider.max_input_tokens)
    return ChromaStore(root, config).search(query, provider, corpus_fingerprint=fingerprint,
                                           mode=mode, top_k=top_k, reference_date=reference_date)
