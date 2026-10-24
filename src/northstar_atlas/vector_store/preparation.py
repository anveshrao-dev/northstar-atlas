"""Verified backend-independent preparation and immutable float32 snapshots."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from time import perf_counter
from typing import Protocol

from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel, EmbeddingProvider, embed_chunks, validate_vector
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import MANIFEST_PATH, PolicyDocument, load_documents
from northstar_atlas.ingestion.refiner import TokenCounter, reconstruct, refine_chunks
from .records import IndexConfig, VectorStoreError, decode_chunk, digest, metadata_for, original_id, validate_context

STORAGE_SCHEMA_VERSION = 1


def load_frozen_snapshot(root: Path, namespace: str) -> "FrozenSnapshot":
    """Read an existing immutable local snapshot; no SDK, model, credentials or writes."""
    try:
        root = Path(root).resolve(strict=True)
        if not isinstance(namespace, str) or not re.fullmatch(r"dev-[0-9a-f]{64}", namespace):
            raise ValueError
        path = root / ".local/pinecone/snapshots" / f"{namespace[4:]}.json"
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError
        data = json.loads(path.read_text(encoding="utf-8"))
        config_data = dict(data["config"])
        model = EmbeddingModel(**config_data.pop("model"))
        config = IndexConfig(model, **config_data)
        records = [EmbeddedChunk(decode_chunk(r["id"], r["text"], r["metadata"], model),
                                 tuple(r["vector"]), model) for r in data["records"]]
        snapshot = freeze_snapshot(records, config, data["corpus_fingerprint"])
        if snapshot.snapshot_id != data["snapshot_id"] or snapshot.namespace != namespace:
            raise ValueError
        return snapshot
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise VectorStoreError("frozen snapshot missing or invalid; no automatic regeneration") from None


class RefiningProvider(EmbeddingProvider, TokenCounter, Protocol):
    """Both existing contracts are needed; no provider inheritance required."""


def corpus_snapshot(root: Path) -> tuple[list[PolicyDocument], str]:
    root = Path(root).resolve(strict=True)
    manifest = root / MANIFEST_PATH
    if not manifest.resolve().is_relative_to(root):
        raise VectorStoreError("manifest resolves outside project")
    before = manifest.read_bytes()
    documents = load_documents(root, mode="draft_review")
    if manifest.read_bytes() != before:
        raise VectorStoreError("manifest changed while loading; retry with a stable corpus")
    fingerprint = digest({"manifest": hashlib.sha256(before).hexdigest(),
                          "sources": {d.document_id: d.content_hash for d in documents}})
    return documents, fingerprint


@dataclass(frozen=True)
class PreparedCorpus:
    records: tuple[EmbeddedChunk, ...]
    config: IndexConfig
    fingerprint: str
    documents: int
    canonical_chunks: int
    max_tokens: int
    warnings: tuple
    embedding_seconds: float


def prepare_corpus(root: Path, provider: RefiningProvider) -> PreparedCorpus:
    """Same loader/refiner/reconstruction checks used by the Chroma workflow."""
    documents, fingerprint = corpus_snapshot(root)
    canonical, warnings = [], []
    for document in documents:
        result = chunk_document(document)
        canonical.extend(result.chunks)
        warnings.extend(result.warnings)
    refined = refine_chunks(canonical, provider).require_complete()
    for original in canonical:
        reconstruct(original, [c for c in refined if original_id(c) == original.chunk_id])
    counts = provider.token_counts([c.text for c in refined])
    if len(counts) != len(refined) or any(n > provider.max_input_tokens for n in counts):
        raise VectorStoreError("refinement exceeds embedding input budget; nothing indexed")
    config = IndexConfig(provider.model, provider.max_input_tokens)
    started = perf_counter()
    records = tuple(embed_chunks(refined, provider))
    elapsed = perf_counter() - started
    if corpus_snapshot(root)[1] != fingerprint:
        raise VectorStoreError("corpus changed during encoding; nothing indexed")
    return PreparedCorpus(records, config, fingerprint, len(documents), len(canonical),
                          max(counts), tuple(warnings), elapsed)


def float32_vector(vector: Sequence[float], config: IndexConfig) -> tuple[float, ...]:
    values = validate_vector(vector, dimension=config.model.dimension)
    try:
        values = struct.unpack(f"<{len(values)}f", struct.pack(f"<{len(values)}f", *values))
    except (OverflowError, struct.error):
        raise VectorStoreError("vector cannot be represented as float32") from None
    validate_vector(values, dimension=config.model.dimension)
    if config.model.normalization != "l2" or not math.isclose(math.hypot(*values), 1, abs_tol=1e-5):
        raise VectorStoreError("Pinecone snapshot requires L2-normalized vectors")
    return values


@dataclass(frozen=True)
class FrozenSnapshot:
    records: tuple[EmbeddedChunk, ...]
    config: IndexConfig
    corpus_fingerprint: str

    @cached_property
    def snapshot_id(self) -> str:
        return digest({"corpus": self.corpus_fingerprint, "config": self.config.serialized,
                       "storage_schema_version": STORAGE_SCHEMA_VERSION,
                       "records": [(r.chunk_id, r.chunk.text, metadata_for(r.chunk, r.model), r.vector)
                                   for r in self.records]})

    @property
    def namespace(self) -> str:
        return "dev-" + self.snapshot_id


def freeze_snapshot(records: Sequence[EmbeddedChunk], config: IndexConfig,
                    corpus_fingerprint: str) -> FrozenSnapshot:
    """Freeze exact records; callers must prepare from verified sources first.

    Sort only the transport snapshot, keeping source order in chunk metadata.
    Freeze float32 values once so repeat writes and backend comparisons agree.
    """
    if not records or not isinstance(corpus_fingerprint, str) or not corpus_fingerprint.strip():
        raise VectorStoreError("snapshot needs records and corpus fingerprint")
    validate_context([r.chunk for r in records], config)
    frozen = []
    for record in records:
        if record.model != config.model:
            raise VectorStoreError("embedding model configuration mismatch")
        metadata_for(record.chunk, record.model)
        frozen.append(replace(record, vector=float32_vector(record.vector, config)))
    return FrozenSnapshot(tuple(sorted(frozen, key=lambda r: r.chunk_id)), config, corpus_fingerprint)
