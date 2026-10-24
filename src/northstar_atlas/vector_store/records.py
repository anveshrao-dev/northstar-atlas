"""Lossless scalar/JSON metadata for Chroma; no model loading or database IO."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from pathlib import Path

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.ingestion.chunker import CHUNKING_VERSION, PolicyChunk
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT
from northstar_atlas.ingestion.refiner import REFINEMENT_VERSION, RefinedPolicyChunk, TextSegment


class VectorStoreError(ValueError):
    """Invalid, stale, incompatible, or incomplete local index."""


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndexConfig:
    model: EmbeddingModel
    token_limit: int
    chunking_version: str = CHUNKING_VERSION
    refinement_version: str = REFINEMENT_VERSION
    schema_version: int = 1
    distance: str = "cosine"

    def __post_init__(self) -> None:
        if not isinstance(self.model, EmbeddingModel) or type(self.token_limit) is not int or self.token_limit < 1:
            raise VectorStoreError("invalid model configuration or token limit")
        if (self.distance != "cosine" or self.schema_version != 1
                or self.chunking_version != CHUNKING_VERSION or self.refinement_version != REFINEMENT_VERSION):
            raise VectorStoreError("unsupported index configuration")

    @property
    def serialized(self) -> str:
        return canonical_json(asdict(self))


def original_id(chunk: PolicyChunk) -> str:
    return chunk.original_chunk_id if isinstance(chunk, RefinedPolicyChunk) else chunk.chunk_id


def context_ids(chunk: PolicyChunk) -> tuple[str, ...]:
    return chunk.required_context_ids if isinstance(chunk, RefinedPolicyChunk) else ()


def metadata_for(chunk: PolicyChunk, model: EmbeddingModel) -> dict[str, str | int]:
    """JSON preserves None/structured fields; scalar columns support filtering."""
    if chunk.status not in (APPROVED, DRAFT):
        raise VectorStoreError("unknown policy approval status")
    if ((chunk.status == APPROVED and not isinstance(chunk.effective_date, date))
            or (chunk.status == DRAFT and chunk.effective_date is not None)):
        raise VectorStoreError("policy status and effective date are inconsistent")
    if not chunk.text.strip() or not chunk.chunk_id:
        raise VectorStoreError("chunk ID and text must be nonempty")
    if chunk.source_path.is_absolute() or ".." in chunk.source_path.parts:
        raise VectorStoreError("source path must remain relative to the project")
    data = asdict(chunk)
    del data["text"]
    data["source_path"] = chunk.source_path.as_posix()
    data["effective_date"] = chunk.effective_date.isoformat() if chunk.effective_date else None
    segments = data.get("segments", [{"source_start": chunk.source_start, "source_end": chunk.source_end,
                "text_start": 0, "text_end": len(chunk.text), "role": "source"}])
    return {
        "document_id": chunk.document_id, "title": chunk.title, "version": chunk.version,
        "department": chunk.department, "status": chunk.status, "section_id": chunk.section_id,
        "source_path": chunk.source_path.as_posix(), "content_hash": chunk.content_hash,
        "effective_date": data["effective_date"] or "",
        "effective_ordinal": chunk.effective_date.toordinal() if chunk.effective_date else 0,
        "original_chunk_id": original_id(chunk), "citation": chunk.citation,
        "embedding_model": model.model_id, "embedding_revision": model.revision or "",
        "embedding_dimension": model.dimension, "embedding_normalization": model.normalization,
        "required_context_ids": canonical_json(context_ids(chunk)),
        "source_spans": canonical_json(segments), "chunk_json": canonical_json(data),
        "kind": "refined" if isinstance(chunk, RefinedPolicyChunk) else "canonical",
    }


def decode_chunk(chunk_id: str, text: str, metadata: dict, model: EmbeddingModel) -> PolicyChunk:
    try:
        data = json.loads(metadata["chunk_json"])
        data["source_path"] = Path(data["source_path"])
        data["effective_date"] = date.fromisoformat(data["effective_date"]) if data["effective_date"] else None
        data["text"] = text
        if metadata["kind"] == "refined":
            data["segments"] = tuple(TextSegment(**s) for s in data["segments"])
            data["required_context_ids"] = tuple(data["required_context_ids"])
            data["embedding_model"] = EmbeddingModel(**data["embedding_model"])
            chunk = RefinedPolicyChunk(**data)
        else:
            chunk = PolicyChunk(**data)
        if chunk.chunk_id != chunk_id or metadata_for(chunk, model) != metadata:
            raise ValueError("metadata mismatch")
    except (ValueError, TypeError, KeyError, AttributeError):
        raise VectorStoreError("record metadata integrity check failed; rerun index") from None
    return chunk


def validate_context(chunks: Sequence[PolicyChunk], config: IndexConfig) -> None:
    """Every refined family must be complete and share source identity/eligibility."""
    by_id = {c.chunk_id: c for c in chunks}
    if len(by_id) != len(chunks):
        raise VectorStoreError("duplicate chunk IDs")
    families: dict[str, list[PolicyChunk]] = {}
    for chunk in chunks:
        families.setdefault(original_id(chunk), []).append(chunk)
    for family in families.values():
        for chunk in family:
            if not isinstance(chunk, RefinedPolicyChunk):
                continue
            if (chunk.embedding_model != config.model or chunk.token_limit != config.token_limit
                    or chunk.refinement_version != config.refinement_version):
                raise VectorStoreError("refined chunk model configuration mismatch")
            required = context_ids(chunk)
            if (len(required) != len(set(required)) or set(required) != {c.chunk_id for c in family} - {chunk.chunk_id}
                    or chunk.refinement_count != len(family)
                    or {getattr(c, "refinement_index", -1) for c in family} != set(range(len(family)))):
                raise VectorStoreError("missing or invalid required sibling context")
            identity = (chunk.document_id, chunk.version, chunk.content_hash, chunk.source_path,
                        chunk.section_id, chunk.status, chunk.effective_date)
            for sibling_id in required:
                sibling = by_id.get(sibling_id)
                if sibling is None or identity != (sibling.document_id, sibling.version, sibling.content_hash,
                        sibling.source_path, sibling.section_id, sibling.status, sibling.effective_date):
                    raise VectorStoreError("required sibling context has incompatible source identity or eligibility")
