"""Local vector storage. Importing this package does not load Chroma or MiniLM."""

from .records import IndexConfig, VectorStoreError
from .store import COLLECTION_NAME, ChromaStore, IndexReport, SearchHit
from .pipeline import corpus_snapshot, index_corpus, search_corpus

__all__ = ["COLLECTION_NAME", "ChromaStore", "IndexConfig", "IndexReport", "SearchHit",
           "VectorStoreError", "corpus_snapshot", "index_corpus", "search_corpus"]
