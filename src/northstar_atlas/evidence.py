"""Immutable evidence values and citation membership checks; no IO or generation."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
import re
from typing import Protocol

from northstar_atlas.ingestion.chunker import PolicyChunk
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from northstar_atlas.ingestion.refiner import TextSegment


class ContextTokenCounter(Protocol):
    """Injected generation-token accounting, not MiniLM's embedding token budget.

    Count the complete supplied string, including any counter-specific wrappers.
    name identifies the tokenizer/revision or explicitly identifies a test counter.
    There is deliberately no approximate default or implicit model download.
    """

    name: str

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class ContextWarning:
    code: str
    message: str
    chunk_ids: tuple[str, ...] = ()


class ContextError(ValueError):
    """Invalid input/critical context: no partial package may be used."""

    def __init__(self, message: str, warnings: tuple[ContextWarning, ...] = ()):
        super().__init__(message)
        self.warnings = warnings


class CitationError(ValueError):
    """Malformed citation or reference absent from the supplied evidence package."""


@dataclass(frozen=True)
class EvidenceSource:
    """Retain the original chunk and exact segment mappings, even after deduplication."""

    chunk: PolicyChunk = field(repr=False)
    segments: tuple[TextSegment, ...]
    required_context_ids: tuple[str, ...]
    direct_rank: int | None


@dataclass(frozen=True)
class EvidenceSpan:
    """One contiguous, exact source excerpt; offsets are end-exclusive characters."""

    source_start: int
    source_end: int
    text: str = field(repr=False)
    chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class CitationReference:
    reference_id: str
    document_id: str
    section_id: str
    version: str
    status: str
    content_hash: str
    chunk_ids: tuple[str, ...]

    @property
    def label(self) -> str:
        if self.status not in (APPROVED, DRAFT):
            raise CitationError("unknown citation approval status")
        authority = "proposed" if self.status == DRAFT else "approved, synthetic"
        return f"{self.document_id} v{self.version} §{self.section_id} ({authority})"


@dataclass(frozen=True)
class EvidenceGroup:
    """An indivisible budget unit for one document snapshot and section."""

    citation: CitationReference
    sources: tuple[EvidenceSource, ...]
    spans: tuple[EvidenceSpan, ...]
    direct_rank: int
    duplicate_characters_removed: int

    @property
    def text(self) -> str:
        """Inspection view. JSON rendering keeps disconnected passages separate."""
        return "\n\n[Source gap: intervening text was not supplied]\n\n".join(s.text for s in self.spans)


@dataclass(frozen=True)
class ContextPackage:
    groups: tuple[EvidenceGroup, ...]
    warnings: tuple[ContextWarning, ...]
    mode: RetrievalMode
    reference_date: date
    rendered: str = field(repr=False)
    token_count: int
    max_tokens: int
    token_counter: str


_CITATION = re.compile(
    r"\[(ev-[0-9a-f]{64})\] [A-Z][A-Z0-9]*-\d{3} "
    r"v\d+(?:\.\d+)* §\d+(?:\.\d+)* \((?:proposed|approved, synthetic)\)"
)


def format_citation(reference: CitationReference) -> str:
    return f"[{reference.reference_id}] {reference.label}"


def citation_syntax_valid(text: str) -> bool:
    """Syntax alone says nothing about source existence, authority or entailment."""
    return isinstance(text, str) and _CITATION.fullmatch(text) is not None


def validate_citations(citations: Sequence[str | CitationReference],
                       package: ContextPackage) -> tuple[CitationReference, ...]:
    """Resolve only exact references supplied after filtering and budgeting.

    Accept structured references or full formatted markers, not bare section names.
    Deduplicate in caller order. This does NOT validate claims or semantic support.
    The package must be the trusted, unchanged result of build_context.
    """
    if isinstance(citations, (str, bytes)) or not isinstance(citations, Sequence):
        raise CitationError("citations must be a sequence of explicit references")
    allowed = {g.citation.reference_id: g.citation for g in package.groups}
    resolved: list[CitationReference] = []
    for citation in citations:
        if isinstance(citation, CitationReference):
            expected = allowed.get(citation.reference_id)
            valid = expected is not None and citation == expected
        elif citation_syntax_valid(citation):
            expected = allowed.get(_CITATION.fullmatch(citation)[1])
            valid = expected is not None and citation == format_citation(expected)
        else:
            raise CitationError("invalid citation syntax")
        if not valid:
            raise CitationError("citation does not match supplied evidence, version and status")
        if expected not in resolved:
            resolved.append(expected)
    return tuple(resolved)
