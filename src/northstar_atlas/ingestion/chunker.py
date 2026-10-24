"""Lossless numbered-section chunking of already verified PolicyDocument values.

Inspect without writing artifacts:
    python -m northstar_atlas.ingestion.chunker --mode draft_review
"""

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import sys

from northstar_atlas.ingestion.document_loader import (
    DRAFT, DocumentLoadError, PolicyDocument, load_documents,
)


DEFAULT_MAX_CHARS = 4_000
CHUNKING_VERSION = "section-v1"
HEADING = re.compile(r"^(#{2,3})[ \t]+(\d+(?:\.\d+)*)\.?[ \t]+(.+?)\s*$")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
STRUCTURED = re.compile(r"(?m)^[ \t]*(?:[-+*]|\d+[.)])[ \t]+|^.*\|.*$|^ {0,3}(?:`{3,}|~{3,})")
HEADER_LABELS = {
    "Organization", "Document ID", "Department", "Policy Owner", "Version",
    "Status", "Effective Date", "Review Cycle", "Classification", "Language", "Draft Date",
}


class ChunkingError(ValueError):
    """A source cannot be chunked without ambiguous citations or lost content."""


@dataclass(frozen=True)
class PolicyChunk:
    """Exact source span plus citation and provenance metadata.

    chunk_index is zero-based within a document. Offsets are Python character
    offsets (end exclusive), not UTF-8 byte offsets. parent_section is an ID.
    """

    chunk_id: str
    document_id: str
    title: str
    department: str
    version: str
    status: str
    effective_date: date | None
    source_path: Path
    content_hash: str
    section_id: str
    section_title: str
    parent_section: str | None
    text: str
    chunk_index: int
    parent_section_title: str | None
    source_start: int
    source_end: int
    classification: str
    language: str
    organization: str
    policy_owner: str
    review_cycle: str
    effective_date_text: str

    @property
    def citation(self) -> str:
        """Render provenance without promoting a proposed draft to authority."""
        label = "proposed" if self.status == DRAFT else "approved, synthetic"
        return f"{self.document_id} v{self.version} §{self.section_id} ({label})"


@dataclass(frozen=True)
class ChunkingResult:
    """Warnings are explicit return data; callers must inspect them before embedding."""

    chunks: tuple[PolicyChunk, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Heading:
    section_id: str
    title: str
    start: int
    body_start: int
    parent_id: str | None
    parent_title: str | None


def _headings(document: PolicyDocument) -> list[_Heading]:
    """Read this corpus's two-level convention; reject ambiguous extensions."""
    headings: list[_Heading] = []
    seen: set[str] = set()
    parent: _Heading | None = None
    fence: str | None = None
    offset = 0
    for line in document.content.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        marker = FENCE.match(stripped)
        if fence is not None:
            if (marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence)
                    and not marker[2].strip()):
                fence = None
        elif marker:
            fence = marker[1]
        elif stripped.startswith("#"):
            match = HEADING.fullmatch(stripped)
            if match:
                level, section_id, title = match.groups()
                if section_id in seen:
                    raise ChunkingError(f"{document.document_id}: duplicate section {section_id}")
                if len(level) == 2:
                    if "." in section_id:
                        raise ChunkingError(f"{document.document_id}: invalid top-level section {section_id}")
                    parent = None
                elif (section_id.count(".") != 1 or parent is None
                      or section_id.split(".")[0] != parent.section_id):
                    raise ChunkingError(f"{document.document_id}: orphan or invalid subsection {section_id}")
                heading = _Heading(
                    section_id, title, offset, offset + len(line),
                    parent.section_id if parent else None, parent.title if parent else None,
                )
                headings.append(heading)
                seen.add(section_id)
                if len(level) == 2:
                    parent = heading
            elif headings or not stripped.startswith("# "):
                raise ChunkingError(f"{document.document_id}: unsupported Markdown heading at character {offset}")
        offset += len(line)
    if fence is not None:
        raise ChunkingError(f"{document.document_id}: unclosed fenced block")
    if not headings:
        raise ChunkingError(f"{document.document_id}: no numbered policy sections")
    # The loader verifies metadata values. This guard prevents silently omitting
    # newly added narrative text before the numbered policy body.
    preamble = document.content[:headings[0].start].splitlines()
    for index, line in enumerate(preamble):
        metadata = re.fullmatch(r"\*\*([^*]+):\*\*[ \t]*(.*)", line)
        if (not line.strip() or (index == 0 and line.startswith("# "))
                or (metadata and metadata[1] in HEADER_LABELS)):
            continue
        raise ChunkingError(f"{document.document_id}: unsupported text before numbered sections")
    return headings


def _atomic_spans(content: str, start: int, body_start: int, end: int) -> list[tuple[int, int]]:
    """Keep structured sections whole; otherwise preserve complete paragraphs.

    A small structural parser cannot reliably decide which prose qualifies a
    table or list. Retaining that entire section is deliberately conservative.
    Its siblings remain independently chunkable. No sentence tokenizer is used.
    """
    body = content[body_start:end]
    if STRUCTURED.search(body):
        return [(start, end)]
    spans: list[tuple[int, int]] = []
    cursor = start
    for separator in re.finditer(r"\r?\n[ \t]*\r?\n(?:[ \t]*\r?\n)*", body):
        boundary = body_start + separator.end()
        if content[max(cursor, body_start):boundary].strip():
            spans.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        if spans and not content[cursor:end].strip():
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((cursor, end))
    return spans or [(start, end)]


def _make_chunk(document: PolicyDocument, heading: _Heading, start: int, end: int,
                index: int) -> PolicyChunk:
    # Span boundaries prevent collisions if another size limit changes the text
    # at the same section/index. Version the algorithm when its semantics change.
    identity = json.dumps(
        [CHUNKING_VERSION, document.document_id, document.version, document.content_hash,
         heading.section_id, start, end, index], separators=(",", ":"),
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return PolicyChunk(
        chunk_id=f"{document.document_id}:{heading.section_id}:{digest}",
        document_id=document.document_id, title=document.title, department=document.department,
        version=document.version, status=document.status, effective_date=document.effective_date,
        source_path=document.source_path, content_hash=document.content_hash,
        section_id=heading.section_id, section_title=heading.title,
        parent_section=heading.parent_id, parent_section_title=heading.parent_title,
        text=document.content[start:end], chunk_index=index, source_start=start, source_end=end,
        classification=document.classification, language=document.language,
        organization=document.organization, policy_owner=document.policy_owner,
        review_cycle=document.review_cycle, effective_date_text=document.effective_date_text,
    )


def chunk_document(document: PolicyDocument, *, max_chars: int = DEFAULT_MAX_CHARS) -> ChunkingResult:
    """Chunk one verified document without IO, filtering, paraphrasing, or overlap.

    max_chars is a soft character limit including source headings/whitespace.
    Atomic blocks exceeding it are retained with explicit warnings. Metadata
    fields do not count toward that limit. Filtering remains the loader's job.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ChunkingError("max_chars must be a positive integer")
    headings = _headings(document)
    chunks: list[PolicyChunk] = []
    warnings: list[str] = []
    pending_start: int | None = None
    for index, heading in enumerate(headings):
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        end = next_heading.start if next_heading else len(document.content)
        start = pending_start if pending_start is not None else heading.start
        pending_start = None
        # Keep a bare parent heading as source context on its first child, rather
        # than producing an evidence-free heading-only embedding candidate.
        if (next_heading and next_heading.parent_id == heading.section_id
                and not document.content[heading.body_start:end].strip()):
            pending_start = start
            continue
        spans = _atomic_spans(document.content, start, heading.body_start, end)
        packed: list[tuple[int, int]] = []
        for block_start, block_end in spans:
            size = block_end - block_start
            if size > max_chars:
                warnings.append(
                    f"{document.document_id} §{heading.section_id}: oversized atomic block "
                    f"[{block_start}:{block_end}], {size} chars > {max_chars}; retained intact"
                )
            if packed and block_end - packed[-1][0] <= max_chars:
                packed[-1] = (packed[-1][0], block_end)
            else:
                packed.append((block_start, block_end))
        for chunk_start, chunk_end in packed:
            chunks.append(_make_chunk(document, heading, chunk_start, chunk_end, len(chunks)))
    return ChunkingResult(tuple(chunks), tuple(warnings))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local numbered-policy chunks; no text bodies or writes.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
    parser.add_argument("--reference-date", help="Inclusive effective-date cutoff, YYYY-MM-DD; default: today UTC.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Soft character limit (default: 4000).")
    args = parser.parse_args()
    if args.max_chars <= 0:
        parser.error("--max-chars must be a positive integer")
    as_of = datetime.now(timezone.utc).date()
    if args.reference_date is not None:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.reference_date):
                raise ValueError
            as_of = date.fromisoformat(args.reference_date)
        except ValueError:
            parser.error("--reference-date must be a valid date in YYYY-MM-DD format")
    try:
        documents = load_documents(args.project_root, mode=args.mode, reference_date=as_of)
        results = [chunk_document(document, max_chars=args.max_chars) for document in documents]
    except (DocumentLoadError, ChunkingError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    sizes = [len(chunk.text) for result in results for chunk in result.chunks]
    print(f"Loaded {len(documents)} document(s); mode={args.mode}; reference_date={as_of}")
    print(f"Total chunks: {len(sizes)}; soft limit: {args.max_chars} characters; algorithm: {CHUNKING_VERSION}")
    if sizes:
        print(f"Characters min/median/max: {min(sizes)}/{median(sizes):g}/{max(sizes)}")
    else:
        print("Characters min/median/max: n/a (no chunks)")
    for document, result in zip(documents, results, strict=True):
        print(f"{document.document_id}: {len(result.chunks)} chunks")
        for chunk in result.chunks:
            print(f"  #{chunk.chunk_index} {chunk.citation}; {len(chunk.text)} chars")
        for warning in result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Oversized-block warnings: {sum(len(result.warnings) for result in results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
