"""Model-aware, lossless refinement AFTER canonical heading-aware chunking.

Refined fragments are embedding units, not independently complete policy rules.
Consumers must honor required_context_ids before presenting a fragment as evidence.
No tokenizer, model, network, or filesystem operation is performed on import.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
import hashlib
import json
import re
from typing import Literal, Protocol

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.ingestion.chunker import PolicyChunk


REFINEMENT_VERSION = "token-context-v1"
# The reviewed semantic bundle below is valid only for this verified source.
PLAT_CONTEXT_HASH = "00e1edff2a7101dc0681b45275b1ae3a684402c0a182e220e8ce1a6d8387d443"
LIST_ITEM = re.compile(r"^ {0,3}(?:[-+*]|\d+[.)])[ \t]+")
HEADING = re.compile(r"^#{2,3}[ \t]+")
ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "fig", "sec", "dept"}


class RefinementError(ValueError):
    """Refinement cannot satisfy its source, budget, or context contract."""


class TokenCounter(Protocol):
    model: EmbeddingModel

    @property
    def max_input_tokens(self) -> int: ...

    def token_counts(self, texts: Sequence[str]) -> list[int]: ...


@dataclass(frozen=True)
class TextSegment:
    """Absolute source offsets and rendered offsets, all end-exclusive characters.

    source segments partition the original once. context segments explicitly
    duplicate original text. Gaps between rendered segments are formatting newlines.
    """

    source_start: int
    source_end: int
    text_start: int
    text_end: int
    role: Literal["source", "context"]


@dataclass(frozen=True)
class RefinedPolicyChunk(PolicyChunk):
    """A PolicyChunk usable by embed_chunks, with explicit refinement provenance.

    Unlike canonical chunks, text may contain duplicated noncontiguous context.
    source_start/end bound only the primary payload; segments describe rendering.
    chunk_index remains the canonical index; refinement_index orders its children.
    """

    original_chunk_id: str
    refinement_index: int
    refinement_count: int
    segments: tuple[TextSegment, ...]
    required_context_ids: tuple[str, ...]
    refinement_version: str
    embedding_model: EmbeddingModel
    token_limit: int


@dataclass(frozen=True)
class RefinementIssue:
    original_chunk_id: str
    citation: str
    reason: str


@dataclass(frozen=True)
class RefinementResult:
    chunks: tuple[PolicyChunk, ...]
    issues: tuple[RefinementIssue, ...]

    def require_complete(self) -> tuple[PolicyChunk, ...]:
        """Refuse downstream encoding if ANY original could not be refined."""
        if self.issues:
            raise RefinementError("; ".join(f"{i.citation}: {i.reason}" for i in self.issues))
        return self.chunks


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    kind: str


@dataclass(frozen=True)
class _Unit:
    start: int
    end: int
    context: tuple[tuple[int, int], ...] = ()


def _blocks(text: str) -> list[_Block]:
    """Recognize this corpus's Markdown without rewriting source characters."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    blocks: list[_Block] = []
    i = 0
    while i < len(lines):
        start = i
        line = lines[i]
        if not line.strip() and blocks:
            blocks[-1] = replace(blocks[-1], end=offsets[i + 1])
            i += 1
            continue
        fence = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence:
            kind = "fence"
            i += 1
            while i < len(lines) and not re.fullmatch(
                    rf" {{0,3}}{re.escape(fence[1][0])}{{{len(fence[1])},}}\s*", lines[i]):
                i += 1
            if i == len(lines):
                raise RefinementError("unsafe unclosed fenced block")
            i += 1
        elif HEADING.match(line):
            kind = "heading"
            i += 1
        elif line.lstrip().startswith("|"):
            kind = "table"
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                i += 1
        else:
            kind = "list" if LIST_ITEM.match(line) else "prose"
            i += 1
            while i < len(lines) and lines[i].strip():
                if (HEADING.match(lines[i]) or lines[i].lstrip().startswith(("|", "```", "~~~"))
                        or (kind == "prose" and LIST_ITEM.match(lines[i]))):
                    break
                i += 1
        blocks.append(_Block(offsets[start], offsets[i], kind))
    return blocks


def _render(chunk: PolicyChunk, unit: _Unit, heading: tuple[int, int]) -> tuple[str, tuple[TextSegment, ...]]:
    """Deduplicate overlapping context, subtract payload, preserve source order."""
    intervals = sorted(set((*unit.context, heading)))
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    spans = [(unit.start, unit.end, "source")]
    for start, end in merged:
        if start < unit.start:
            spans.append((start, min(end, unit.start), "context"))
        if end > unit.end:
            spans.append((max(start, unit.end), end, "context"))
    text = ""
    segments: list[TextSegment] = []
    previous_end = None
    for start, end, role in sorted(spans):
        part = chunk.text[start:end]
        if text and previous_end != start:
            # A repeated table header must directly precede its row, not be
            # separated by a blank line (which would end a Markdown table).
            table_join = text.rstrip().endswith("|") and part.lstrip().startswith("|")
            needed = 1 if table_join else 2
            present = len(text) - len(text.rstrip("\n"))
            text += "\n" * max(0, needed - present)
        offset = len(text)
        text += part
        segments.append(TextSegment(chunk.source_start + start, chunk.source_start + end,
                                    offset, len(text), role))
        previous_end = end
    return text, tuple(segments)


def _refine_one(chunk: PolicyChunk, counter: TokenCounter) -> tuple[RefinedPolicyChunk, ...]:
    blocks = _blocks(chunk.text)
    headings = [b for b in blocks if b.kind == "heading"]
    if not headings:
        # Canonical long-section continuations may have heading metadata only.
        # Do not invent heading text; the citation fields remain available.
        heading = (0, 0)
    else:
        heading = (headings[-1].start, headings[-1].end)

    def fits(unit: _Unit) -> bool:
        return counter.token_counts([_render(chunk, unit, heading)[0]])[0] <= counter.max_input_tokens

    protected: dict[int, _Unit] = {}
    if chunk.document_id == "PLAT-001" and chunk.section_id == "6.1":
        if chunk.content_hash != PLAT_CONTEXT_HASH:
            raise RefinementError("unsafe to reuse PLAT §6.1 context recipe: changed source requires review")
        try:
            approval = next(b for b in blocks if chunk.text[b.start:b.end].startswith("Record technical approval"))
            drift = next(b for b in blocks if chunk.text[b.start:b.end].startswith("If Terraform detects drift"))
            conditions = next(b for b in blocks if chunk.text[b.start:b.end].startswith("No separate Change Management approval"))
        except StopIteration:
            raise RefinementError("unsafe PLAT §6.1 context recipe mismatch; review required") from None
        if drift.end != conditions.start:
            raise RefinementError("unsafe PLAT §6.1 context adjacency changed; review required")
        protected[drift.start] = _Unit(drift.start, conditions.end, ((approval.start, approval.end),))

    units: list[_Unit] = []
    skip_until = 0
    for index, block in enumerate(blocks):
        if block.start < skip_until:
            continue
        if block.start in protected:
            unit = protected[block.start]
            if not fits(unit):
                raise RefinementError("unsafe PLAT §6.1 approval/scope context exceeds token budget; not split")
            units.append(unit)
            skip_until = unit.end
            continue
        prior = blocks[index - 1] if index else None
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        context = ((prior.start, prior.end),) if prior and prior.kind == "prose" and block.kind in {"table", "list"} else ()
        whole = _Unit(block.start, block.end, context)
        if block.kind == "table":
            lines = chunk.text[block.start:block.end].splitlines(keepends=True)
            if len(lines) < 3 or not all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell)
                                       for cell in lines[1].strip().strip("|").split("|")):
                raise RefinementError("unsafe or unsupported Markdown table header")
            header_end = block.start + len(lines[0]) + len(lines[1])
            row_starts = [header_end]
            for line in lines[2:]:
                if line.strip():
                    row_starts.append(row_starts[-1] + len(line))
            row_ends = row_starts[1:]
            row_starts = row_starts[:-1]
            if not row_ends:
                raise RefinementError("unsafe header-only table has no data rows")
            row_ends[-1] = block.end  # preserve trailing blank lines exactly
            table_context = (*context, (block.start, header_end))
            rows = [_Unit(block.start if i == 0 else start, end, table_context)
                    for i, (start, end) in enumerate(zip(row_starts, row_ends, strict=True))]
            # Repeat the adjacent qualification when every row can carry it.
            # Remaining section context is mandatory via sibling IDs, never lost.
            if following and following.kind == "prose":
                qualified = [replace(u, context=(*u.context, (following.start, following.end))) for u in rows]
                if all(fits(u) for u in qualified):
                    rows = qualified
            if not all(fits(u) for u in rows):
                raise RefinementError(f"unsafe table row plus header/context at source {chunk.source_start + block.start}; cannot split a row")
            units.extend(rows)
        elif fits(whole):
            units.append(whole)
        elif block.kind == "list":
            body = chunk.text[block.start:block.end]
            indent = len(body) - len(body.lstrip(" "))
            # Nested items remain attached to their outer item. Only siblings
            # at the block's starting indentation are candidate boundaries.
            starts = [block.start + m.start() for m in re.finditer(
                rf"(?m)^ {{{indent}}}(?:[-+*]|\d+[.)])[ \t]+", body)]
            items = [_Unit(start, end, context) for start, end in zip(starts, [*starts[1:], block.end], strict=True)]
            if not items or not all(fits(u) for u in items):
                raise RefinementError("unsafe list item plus governing context exceeds budget; item not split")
            units.extend(items)
        elif block.kind == "prose":
            # Conservative English sentence boundaries; titles, initials and
            # dotted abbreviations are ambiguous, so do not split there.
            body = chunk.text[block.start:block.end]
            ends = []
            for match in re.finditer(r"[.!?][ \t\r\n]+(?=[A-Z*])", body):
                previous = re.search(r"([\w.]+)$", body[:match.start()])
                word = previous[1] if previous else ""
                if body[match.start()] == "." and (len(word) <= 1 or "." in word or word.lower() in ABBREVIATIONS):
                    continue
                ends.append(block.start + match.end())
            ends.append(block.end)
            sentences = [_Unit(start, end) for start, end in zip([block.start, *ends[:-1]], ends, strict=True)]
            if not all(fits(u) for u in sentences):
                raise RefinementError("unsafe oversized sentence or unrecognized boundary; no forced split")
            units.extend(sentences)
        else:
            raise RefinementError(f"unsafe oversized {block.kind} block; no forced split")

    packed: list[_Unit] = []
    for unit in units:
        if packed:
            candidate = _Unit(packed[-1].start, unit.end, (*packed[-1].context, *unit.context))
            if fits(candidate):
                packed[-1] = candidate
                continue
        packed.append(unit)
    refined: list[RefinedPolicyChunk] = []
    for index, unit in enumerate(packed):
        text, segments = _render(chunk, unit, heading)
        if not fits(unit):
            raise RefinementError("unsafe fragment exceeds input budget")
        identity = [REFINEMENT_VERSION, chunk.chunk_id, chunk.content_hash,
                    counter.model.model_id, counter.model.revision, counter.max_input_tokens,
                    index, unit.start, unit.end, [(s.source_start, s.source_end, s.role) for s in segments], text]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        values = {f.name: getattr(chunk, f.name) for f in fields(PolicyChunk)}
        values.update(chunk_id=f"refined:{REFINEMENT_VERSION}:{digest}", text=text,
                      source_start=chunk.source_start + unit.start, source_end=chunk.source_start + unit.end)
        refined.append(RefinedPolicyChunk(**values, original_chunk_id=chunk.chunk_id,
                       refinement_index=index, refinement_count=len(packed), segments=segments,
                       required_context_ids=(), refinement_version=REFINEMENT_VERSION,
                       embedding_model=counter.model, token_limit=counter.max_input_tokens))
    return tuple(replace(c, required_context_ids=tuple(p.chunk_id for p in refined if p.chunk_id != c.chunk_id))
                 for c in refined)


def reconstruct(original: PolicyChunk, pieces: Sequence[PolicyChunk]) -> str:
    """Validate and reconstruct from rendered PRIMARY spans, excluding duplicates.

    This proves character coverage, not semantic completeness of a single piece.
    Missing, overlapping, duplicated, out-of-range, or altered text is an error.
    """
    if len(pieces) == 1 and pieces[0] == original:
        return pieces[0].text
    primary: list[tuple[int, int, str]] = []
    for piece in pieces:
        if not isinstance(piece, RefinedPolicyChunk) or piece.original_chunk_id != original.chunk_id:
            raise RefinementError("reconstruction contains an unrelated fragment")
        rendered_end = 0
        for segment in piece.segments:
            if not (original.source_start <= segment.source_start <= segment.source_end <= original.source_end
                    and rendered_end <= segment.text_start <= segment.text_end <= len(piece.text)):
                raise RefinementError("invalid source or rendered offsets")
            if piece.text[rendered_end:segment.text_start].strip("\n"):
                raise RefinementError("untracked rendered content")
            text = piece.text[segment.text_start:segment.text_end]
            expected = original.text[segment.source_start-original.source_start:segment.source_end-original.source_start]
            if text != expected:
                raise RefinementError("rendered source/context was altered")
            if segment.role == "source":
                primary.append((segment.source_start, segment.source_end, text))
            elif segment.role != "context":
                raise RefinementError("unrecognized segment role")
            rendered_end = segment.text_end
        if rendered_end != len(piece.text):
            raise RefinementError("untracked trailing content")
    cursor = original.source_start
    reconstructed = ""
    for start, end, text in sorted(primary):
        if start != cursor:
            raise RefinementError("source coverage has a gap, duplicate, or overlap")
        reconstructed += text
        cursor = end
    if cursor != original.source_end or reconstructed != original.text:
        raise RefinementError("source reconstruction is incomplete")
    return reconstructed


def refine_chunks(chunks: Sequence[PolicyChunk], counter: TokenCounter) -> RefinementResult:
    """Refine only oversized canonical inputs; retain unsafe originals as issues.

    Use require_complete() before embedding. No partial fragments of a failed
    original are emitted. Original objects and the canonical chunker are untouched.
    """
    if type(counter.max_input_tokens) is not int or counter.max_input_tokens <= 0:
        raise RefinementError("token limit must be a positive integer")
    originals = tuple(chunks)
    if any(not isinstance(c, PolicyChunk) or isinstance(c, RefinedPolicyChunk) for c in originals):
        raise RefinementError("refinement expects canonical PolicyChunk objects")
    if len({c.chunk_id for c in originals}) != len(originals):
        raise RefinementError("duplicate canonical chunk IDs")
    if any(not c.text.strip() or len(c.text) != c.source_end - c.source_start for c in originals):
        raise RefinementError("canonical chunk text/span mismatch")
    counts = counter.token_counts([c.text for c in originals])
    if len(counts) != len(originals) or any(type(n) is not int or n <= 0 for n in counts):
        raise RefinementError("invalid tokenizer counts")
    output: list[PolicyChunk] = []
    issues: list[RefinementIssue] = []
    for original, count in zip(originals, counts, strict=True):
        if count <= counter.max_input_tokens:
            output.append(original)
            continue
        try:
            pieces = _refine_one(original, counter)
            reconstruct(original, pieces)
        except RefinementError as exc:
            issues.append(RefinementIssue(original.chunk_id, original.citation, str(exc)))
        else:
            output.extend(pieces)
    return RefinementResult(tuple(output), tuple(issues))


def main() -> int:
    """Inspect refinement offline; encode only explicitly requested samples."""
    import argparse
    from collections import Counter
    from datetime import date
    import math
    from pathlib import Path
    from statistics import median
    import sys
    from time import perf_counter

    from northstar_atlas.embeddings import EmbeddingError, embed_chunks, embed_query
    from northstar_atlas.ingestion.chunker import ChunkingError, chunk_document
    from northstar_atlas.ingestion.document_loader import DocumentLoadError, load_documents
    from northstar_atlas.local_embeddings import MiniLMProvider

    parser = argparse.ArgumentParser(description="Inspect token refinement using the cached model; no downloads or vector writes.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
    parser.add_argument("--reference-date", type=date.fromisoformat)
    parser.add_argument("--section", help="Filter displayed/sample chunks, e.g. PLAT-001:6.1; corpus scan remains complete.")
    parser.add_argument("--show-text", action="store_true", help="Print selected section text and segment provenance.")
    parser.add_argument("--samples", type=int, default=0, help="Number of selected chunks to embed, plus one query (default: 0).")
    args = parser.parse_args()
    if args.samples < 0 or (args.show_text and not args.section):
        parser.error("--samples must be nonnegative; --show-text requires --section")
    try:
        documents = load_documents(args.project_root, mode=args.mode, reference_date=args.reference_date)
        originals = tuple(c for d in documents for c in chunk_document(d).chunks)
        if not originals:
            raise RefinementError("no policy chunks available in requested mode/date")
        provider = MiniLMProvider(args.project_root)
        old_counts = provider.token_counts([c.text for c in originals])
        started = perf_counter()
        result = refine_chunks(originals, provider)
        refinement_seconds = perf_counter() - started
        print(f"Model: {provider.model.model_id}; revision: {provider.model.revision}")
        print(f"Documents: {len(documents)}; mode: {args.mode}; Original chunks: {len(originals)}")
        print(f"Refined output chunks: {len(result.chunks)}; Unsafe originals: {len(result.issues)}")
        oversized = sum(n > provider.max_input_tokens for n in old_counts)
        print(f"Oversized originals resolved: {oversized-len(result.issues)}/{oversized}")
        for issue in result.issues:
            print(f"UNSAFE: {issue.citation}: {issue.reason}", file=sys.stderr)
        chunks = result.require_complete()  # No encoding on a partial corpus.
        counts = provider.check_inputs([c.text for c in chunks])
        print(f"Tokens min/median/max (including special tokens): {min(counts)}/{median(counts):g}/{max(counts)}")
        print("Remaining oversized outputs: 0")
        old_by_doc = Counter(c.document_id for c in originals)
        new_by_doc = Counter(c.document_id for c in chunks)
        for doc, n in old_by_doc.items():
            print(f"{doc}: {n} canonical -> {new_by_doc[doc]} embedding chunks")
        refined = [c for c in chunks if isinstance(c, RefinedPolicyChunk)]
        print(f"Fragments requiring sibling context before standalone evidence use: {len(refined)}")
        selected = [c for c in chunks if args.section is None or f"{c.document_id}:{c.section_id}" == args.section]
        if not selected:
            raise RefinementError("requested section is absent in this mode/date")
        for chunk in selected if args.section else ():
            print(f"{chunk.citation}; tokens={provider.token_counts([chunk.text])[0]}; ID={chunk.chunk_id}")
            if isinstance(chunk, RefinedPolicyChunk):
                print(f"  original={chunk.original_chunk_id}; part={chunk.refinement_index+1}/{chunk.refinement_count}; "
                      f"required_context_ids={','.join(chunk.required_context_ids)}")
                if args.show_text:
                    for segment in chunk.segments:
                        print(f"  {segment.role}: source[{segment.source_start}:{segment.source_end}] "
                              f"-> text[{segment.text_start}:{segment.text_end}]")
            if args.show_text:
                print(chunk.text)
        if args.samples:
            started = perf_counter()
            embedded = embed_chunks(selected[:args.samples], provider)
            document_seconds = perf_counter() - started
            started = perf_counter()
            query = embed_query("When does a regenerated Terraform plan require Change Management approval?", provider)
            query_seconds = perf_counter() - started
            for item in embedded:
                print(f"Vector {item.chunk_id}: dimensions={len(item.vector)}; norm={math.hypot(*item.vector):.6f}; finite=True")
            print(f"Encoded samples: {len(embedded)}; query vectors: 1; dimension: {len(query.vector)}; "
                  f"query norm={math.hypot(*query.vector):.6f}")
            print(f"Embedding seconds: documents={document_seconds:.3f}; query={query_seconds:.3f}")
        print(f"Provider initialization seconds: {provider.load_seconds:.3f}; refinement seconds: {refinement_seconds:.3f}")
        return 0
    except (RefinementError, EmbeddingError, DocumentLoadError, ChunkingError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
