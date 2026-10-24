"""Provider-independent embedding contracts and validation; standard library only.

This module does not load a model, perform network IO, or persist vectors.
Use a verified PolicyChunk from the existing loader/chunker pipeline.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Literal, Protocol

from northstar_atlas.ingestion.chunker import PolicyChunk


class EmbeddingError(ValueError):
    """Invalid input, model metadata, or provider output; no partial result returned."""


class ProviderError(EmbeddingError):
    """An encoding operation failed without exposing provider response contents."""


def _validate_dimension(dimension: int) -> None:
    if type(dimension) is not int or dimension <= 0:
        raise EmbeddingError("embedding dimension must be a positive integer")


@dataclass(frozen=True)
class EmbeddingModel:
    """Identity of the vector space, not the policy's version or source hash.

    revision=None explicitly means unavailable, not a fabricated immutable pin.
    normalization describes returned vectors; this module never changes it.
    """

    model_id: str
    revision: str | None
    dimension: int
    normalization: Literal["none", "l2"] = "none"

    def __post_init__(self) -> None:
        _validate_dimension(self.dimension)
        for label, value in (("model_id", self.model_id), ("revision", self.revision)):
            if label == "revision" and value is None:
                continue
            if (not isinstance(value, str) or not value.strip() or value != value.strip()
                    or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise EmbeddingError(f"{label} must be a nonempty single-line string")
        if self.normalization not in ("none", "l2"):
            raise EmbeddingError("normalization must be 'none' or 'l2'")


class EmbeddingProvider(Protocol):
    """A small structural interface: adapters need no base-class inheritance.

    Document results MUST preserve input order, including when internally batched.
    Adapters must return Python numeric sequences, handle model-specific prompts
    and pooling consistently, and reject over-limit input rather than truncate it.
    Query/document methods are separate because some models encode them differently.
    """

    @property
    def model(self) -> EmbeddingModel: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


@dataclass(frozen=True)
class EmbeddedChunk:
    """Retain the complete immutable chunk, so no source metadata is dropped."""

    chunk: PolicyChunk = field(repr=False)
    vector: tuple[float, ...] = field(repr=False)
    model: EmbeddingModel

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass(frozen=True)
class EmbeddedQuery:
    """A query vector with matching model provenance; query text is not retained."""

    vector: tuple[float, ...] = field(repr=False)
    model: EmbeddingModel


def validate_vector(vector: Sequence[float], *, dimension: int) -> tuple[float, ...]:
    """Copy a finite, nonzero vector of the declared dimension into a tuple.

    Booleans and numeric strings are not vector coordinates. A zero vector has
    undefined cosine similarity. An overflowing norm cannot be compared reliably.
    """
    _validate_dimension(dimension)
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise EmbeddingError("vector must be a numeric sequence")
    if len(vector) != dimension:
        raise EmbeddingError(f"vector dimension mismatch: expected {dimension}, received {len(vector)}")
    values: list[float] = []
    for index, value in enumerate(vector):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingError(f"vector coordinate {index} must be a real number")
        try:
            number = float(value)
        except (OverflowError, ValueError, TypeError):
            raise EmbeddingError(f"vector coordinate {index} cannot be represented as a float") from None
        if not math.isfinite(number):
            raise EmbeddingError(f"vector coordinate {index} must be finite")
        values.append(number)
    norm = math.hypot(*values)
    if norm == 0 or not math.isfinite(norm):
        raise EmbeddingError("vector must have a finite, nonzero norm")
    return tuple(values)


def _provider_model(provider: EmbeddingProvider) -> EmbeddingModel:
    try:
        model = provider.model
    except Exception as exc:
        raise ProviderError(f"embedding provider metadata failed ({type(exc).__name__})") from None
    if not isinstance(model, EmbeddingModel):
        raise EmbeddingError("provider model must be an EmbeddingModel")
    return model


def _validated_output(vector: Sequence[float], model: EmbeddingModel, label: str) -> tuple[float, ...]:
    try:
        values = validate_vector(vector, dimension=model.dimension)
        if model.normalization == "l2" and not math.isclose(math.hypot(*values), 1.0, rel_tol=1e-5):
            raise EmbeddingError("vector does not match declared l2 normalization")
    except EmbeddingError as exc:
        raise EmbeddingError(f"{label}: {exc}") from None
    return values


def embed_chunks(chunks: Sequence[PolicyChunk], provider: EmbeddingProvider) -> list[EmbeddedChunk]:
    """Return one validated vector per chunk, preserving order and metadata.

    The provider's ordering contract is necessary: numerical values alone cannot
    reveal a provider that silently reorders its results. No automatic retries,
    partial returns, truncation, normalization, or storage operations occur here.
    """
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes, bytearray)):
        raise EmbeddingError("chunks must be a sequence of PolicyChunk objects")
    snapshot = tuple(chunks)
    seen: set[str] = set()
    for index, chunk in enumerate(snapshot):
        if not isinstance(chunk, PolicyChunk):
            raise EmbeddingError(f"chunk {index} must be a PolicyChunk")
        if not isinstance(chunk.text, str) or not chunk.text.strip():
            raise EmbeddingError(f"chunk {index} text must be a nonempty string")
        if not isinstance(chunk.chunk_id, str) or not chunk.chunk_id.strip():
            raise EmbeddingError(f"chunk {index} ID must be a nonempty string")
        if chunk.chunk_id in seen:
            raise EmbeddingError(f"duplicate chunk ID at position {index}")
        seen.add(chunk.chunk_id)
    model = _provider_model(provider)
    if not snapshot:
        return []
    try:
        vectors = provider.embed_documents(tuple(chunk.text for chunk in snapshot))
    except Exception as exc:
        raise ProviderError(f"document embedding provider failed ({type(exc).__name__})") from None
    if _provider_model(provider) != model:
        raise EmbeddingError("provider model changed during document embedding")
    if not isinstance(vectors, Sequence) or isinstance(vectors, (str, bytes, bytearray)):
        raise EmbeddingError("provider document output must be a sequence of vectors")
    if len(vectors) != len(snapshot):
        raise EmbeddingError(f"provider vector count mismatch: expected {len(snapshot)}, received {len(vectors)}")
    return [
        EmbeddedChunk(chunk, _validated_output(vector, model, f"document vector {index}"), model)
        for index, (chunk, vector) in enumerate(zip(snapshot, vectors, strict=True))
    ]


def embed_query(text: str, provider: EmbeddingProvider) -> EmbeddedQuery:
    """Encode a nonempty query through the same declared vector space."""
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("query must be a nonempty string")
    model = _provider_model(provider)
    try:
        vector = provider.embed_query(text)
    except Exception as exc:
        raise ProviderError(f"query embedding provider failed ({type(exc).__name__})") from None
    if _provider_model(provider) != model:
        raise EmbeddingError("provider model changed during query embedding")
    return EmbeddedQuery(_validated_output(vector, model, "query vector"), model)
