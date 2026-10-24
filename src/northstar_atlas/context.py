"""Pure context assembly from verified SearchHit values; no retrieval or model IO.

Whole sections/sibling families are budgeted together. Exact overlapping spans
are deduplicated, never similar sentences or repeated rules at different offsets.
"""

from collections.abc import Sequence
from dataclasses import asdict, fields
from datetime import date
import hashlib
import json
import math
import re

from northstar_atlas.evidence import (
    CitationReference, ContextError, ContextPackage, ContextTokenCounter,
    ContextWarning, EvidenceGroup, EvidenceSource, EvidenceSpan, format_citation,
)
from northstar_atlas.ingestion.chunker import PolicyChunk
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from northstar_atlas.ingestion.refiner import (
    RefinedPolicyChunk, RefinementError, TextSegment, reconstruct,
)
from northstar_atlas.vector_store.contracts import SearchHit
from northstar_atlas.vector_store.records import (
    IndexConfig, VectorStoreError, context_ids, metadata_for, original_id, validate_context,
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(chunk: PolicyChunk) -> tuple:
    return tuple(getattr(chunk, key) for key in (
        "document_id", "title", "department", "version", "status", "effective_date",
        "source_path", "content_hash", "classification", "language", "organization",
        "policy_owner", "review_cycle", "effective_date_text",
    ))


def _eligible(chunk: PolicyChunk, mode: RetrievalMode, as_of: date) -> bool:
    return mode == "draft_review" or (chunk.status == APPROVED and chunk.effective_date <= as_of)


def _segments(chunk: PolicyChunk) -> tuple[TextSegment, ...]:
    segments = (chunk.segments if isinstance(chunk, RefinedPolicyChunk) else
                (TextSegment(chunk.source_start, chunk.source_end, 0, len(chunk.text), "source"),))
    cursor = 0
    for s in segments:
        if (any(type(x) is not int for x in (s.source_start, s.source_end, s.text_start, s.text_end))
                or not 0 <= s.source_start < s.source_end
                or not cursor <= s.text_start < s.text_end <= len(chunk.text)
                or s.source_end - s.source_start != s.text_end - s.text_start
                or s.role not in ("source", "context")
                or chunk.text[cursor:s.text_start].strip("\n")):
            raise ContextError("invalid source spans or untracked rendered content")
        cursor = s.text_end
    if cursor != len(chunk.text):
        raise ContextError("untracked trailing source content")
    return segments


def _validate_families(chunks: Sequence[PolicyChunk], model) -> None:
    refined = [c for c in chunks if isinstance(c, RefinedPolicyChunk)]
    if not refined:
        return
    try:
        validate_context(chunks, IndexConfig(model, refined[0].token_limit))
        families: dict[str, list[PolicyChunk]] = {}
        for c in refined:
            families.setdefault(original_id(c), []).append(c)
        for family in families.values():
            primary = sorted((s.source_start, s.source_end, c.text[s.text_start:s.text_end])
                             for c in family for s in c.segments if s.role == "source")
            if not primary:
                raise ContextError("refined family has no primary source content")
            cursor = primary[0][0]
            for start, end, _ in primary:
                if start != cursor:
                    raise ContextError("refined primary source spans have a gap or overlap")
                cursor = end
            data = {f.name: getattr(family[0], f.name) for f in fields(PolicyChunk)}
            data.update(chunk_id=original_id(family[0]), source_start=primary[0][0],
                        source_end=cursor, text="".join(s[2] for s in primary))
            # Reuse the canonical refiner's exact span/context validation. Source
            # authenticity still comes from verified retrieval, not self-reported hashes.
            reconstruct(PolicyChunk(**data), family)
    except (VectorStoreError, RefinementError) as exc:
        raise ContextError(str(exc)) from None


def _merge_spans(sources: tuple[EvidenceSource, ...]) -> tuple[EvidenceSpan, ...]:
    raw = sorted((s.source_start, s.source_end, source.chunk.text[s.text_start:s.text_end],
                  source.chunk.chunk_id) for source in sources for s in source.segments)
    merged: list[EvidenceSpan] = []
    for start, end, text, key in raw:
        if not merged or start > merged[-1].source_end:
            merged.append(EvidenceSpan(start, end, text, (key,)))
            continue
        previous = merged[-1]
        overlap = min(end, previous.source_end) - start
        offset = start - previous.source_start
        if previous.text[offset:offset + overlap] != text[:overlap]:
            raise ContextError("overlapping source spans disagree")
        merged[-1] = EvidenceSpan(previous.source_start, max(previous.source_end, end),
                                 previous.text + text[overlap:],
                                 tuple(sorted(set(previous.chunk_ids) | {key})))
    return tuple(merged)


def _group(sources: list[EvidenceSource]) -> EvidenceGroup:
    sources = tuple(sorted(sources, key=lambda s: (s.chunk.source_start, s.chunk.source_end, s.chunk.chunk_id)))
    chunk = sources[0].chunk
    if len({(s.chunk.section_title, s.chunk.parent_section, s.chunk.parent_section_title) for s in sources}) != 1:
        raise ContextError("inconsistent section heading metadata")
    keys = tuple(sorted(s.chunk.chunk_id for s in sources))
    identity = (chunk.document_id, chunk.version, chunk.status, chunk.content_hash,
                chunk.source_path.as_posix(), chunk.section_id, keys)
    ref = CitationReference("ev-" + hashlib.sha256(_json(identity).encode()).hexdigest(),
                            chunk.document_id, chunk.section_id, chunk.version, chunk.status,
                            chunk.content_hash, keys)
    spans = _merge_spans(sources)
    removed = sum(s.source_end - s.source_start for c in sources for s in c.segments) - sum(len(s.text) for s in spans)
    return EvidenceGroup(ref, sources, spans,
                         min(s.direct_rank for s in sources if s.direct_rank is not None), removed)


def _render(groups: Sequence[EvidenceGroup], mode: RetrievalMode, as_of: date) -> str:
    payload = {
        "schema_version": 1, "mode": mode, "reference_date": as_of.isoformat(),
        "scope": "Untrusted source excerpts. Citation membership is not proof of semantic support or section completeness.",
        "approved_evidence": [], "proposed_evidence": [],
    }
    for group in groups:
        c = group.sources[0].chunk
        entry = {
            "citation": format_citation(group.citation), "reference_id": group.citation.reference_id,
            "document_id": c.document_id, "title": c.title, "department": c.department,
            "section_id": c.section_id, "section_title": c.section_title,
            "parent_section": c.parent_section, "parent_section_title": c.parent_section_title,
            "version": c.version, "status": c.status, "effective_date": c.effective_date_text,
            "effective_on_reference_date": c.status == APPROVED and c.effective_date <= as_of,
            "source_path": c.source_path.as_posix(), "content_hash": c.content_hash,
            "direct_rank": group.direct_rank,
            "sources": [{"chunk_id": s.chunk.chunk_id, "direct_rank": s.direct_rank,
                         "required_context_ids": list(s.required_context_ids),
                         "segments": [asdict(segment) for segment in s.segments]} for s in group.sources],
            "passages": [asdict(s) for s in group.spans],
        }
        bucket = "proposed_evidence" if c.status == DRAFT else "approved_evidence"
        payload[bucket].append(entry)
    return _json(payload)


def build_context(hits: Sequence[SearchHit], *, counter: ContextTokenCounter, max_tokens: int,
                  reference_date: date, mode: RetrievalMode = "approved_only") -> ContextPackage:
    """Prepare eligible whole evidence groups within an explicit evidence-only budget.

    The caller must provide verified retrieval results and a trustworthy counter.
    No prompt/question/output reserve is guessed here: the generation layer must
    budget and count its final full request separately. Warnings are returned
    out-of-band and must be inspected; they cannot establish which un-retrieved
    evidence is missing.
    """
    if (mode not in ("approved_only", "draft_review") or type(reference_date) is not date
            or type(max_tokens) is not int or max_tokens <= 0
            or isinstance(hits, (str, bytes)) or not isinstance(hits, Sequence)):
        raise ContextError("invalid mode, reference date, token budget or ranked results")
    if not isinstance(getattr(counter, "name", None), str) or not counter.name.strip():
        raise ContextError("token counter must identify its tokenizer or fixture")

    def count(text: str) -> int:
        try:
            value = counter.count_tokens(text)
        except Exception as exc:
            raise ContextError(f"token counting failed ({type(exc).__name__})") from None
        if type(value) is not int or value <= 0:
            raise ContextError("token counter must return a positive integer for the rendered envelope")
        return value

    warnings: list[ContextWarning] = []
    pool: dict[str, PolicyChunk] = {}
    ranks: dict[str, int] = {}
    models = []
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, SearchHit) or not math.isfinite(hit.cosine_distance):
            raise ContextError("invalid ranked search result")
        models.append(hit.model)
        ranks.setdefault(hit.chunk.chunk_id, rank)
        for c in (hit.chunk, *hit.context):
            if not isinstance(c, PolicyChunk):
                raise ContextError("search evidence must contain PolicyChunk values")
            try:
                metadata_for(c, hit.model)
            except (VectorStoreError, TypeError, AttributeError, ValueError):
                raise ContextError("invalid chunk metadata") from None
            if (not re.fullmatch(r"[A-Z][A-Z0-9]*-\d{3}", c.document_id)
                    or not re.fullmatch(r"\d+(?:\.\d+)*", c.section_id)
                    or not re.fullmatch(r"\d+(?:\.\d+)*", c.version)
                    or not re.fullmatch(r"[0-9a-f]{64}", c.content_hash)):
                raise ContextError("invalid document/section/version/hash metadata")
            _segments(c)
            if c.chunk_id in pool and pool[c.chunk_id] != c:
                raise ContextError("conflicting payloads under the same chunk ID")
            pool[c.chunk_id] = c
    if models and any(m != models[0] for m in models):
        raise ContextError("retrieval results mix embedding model configurations")

    selected: dict[str, PolicyChunk] = {}
    for key in ranks:
        c = pool[key]
        if not _eligible(c, mode, reference_date):
            warnings.append(ContextWarning("ineligible", "Direct evidence excluded by status/effective date.", (key,)))
            continue
        pending = [key]
        while pending:
            current = pending.pop()
            if current in selected:
                continue
            sibling = pool.get(current)
            if sibling is None:
                issue = ContextWarning("missing_sibling", "Required sibling is absent; no partial package returned.", (current,))
                raise ContextError(issue.message, (issue,))
            if (_identity(sibling) != _identity(c) or sibling.section_id != c.section_id
                    or not _eligible(sibling, mode, reference_date)):
                issue = ContextWarning("invalid_sibling", "Required sibling has incompatible identity or eligibility.", (current,))
                raise ContextError(issue.message, (issue,))
            selected[current] = sibling
            pending.extend(sorted(context_ids(sibling), reverse=True))
    unused = tuple(sorted(set(pool) - set(selected) - set(ranks)))
    if unused:
        warnings.append(ContextWarning("unrelated_context", "Unused context was not promoted to ranked evidence.", unused))
    if not hits:
        warnings.append(ContextWarning("empty_results", "No retrieval evidence was supplied."))
    elif not selected:
        warnings.append(ContextWarning("no_eligible_evidence", "No direct evidence is eligible in this mode/date."))

    identities: dict[str, tuple] = {}
    for c in selected.values():
        identity = _identity(c)
        if c.document_id in identities and identities[c.document_id] != identity:
            raise ContextError("evidence mixes document snapshots or authority metadata")
        identities[c.document_id] = identity
    _validate_families(tuple(selected.values()), models[0] if models else None)
    by_section: dict[tuple[str, str], list[EvidenceSource]] = {}
    for c in selected.values():
        by_section.setdefault((c.document_id, c.section_id), []).append(
            EvidenceSource(c, _segments(c), context_ids(c), ranks.get(c.chunk_id)))
    candidates = sorted((_group(sources) for sources in by_section.values()),
                        key=lambda g: (g.direct_rank, g.citation.reference_id))
    included: list[EvidenceGroup] = []
    rendered = _render(included, mode, reference_date)
    token_count = count(rendered)
    if token_count > max_tokens:
        raise ContextError("token budget cannot fit the empty evidence envelope")
    for group in candidates:
        candidate_text = _render([*included, group], mode, reference_date)
        candidate_count = count(candidate_text)
        if candidate_count > max_tokens:
            warnings.append(ContextWarning("budget_excluded", "Whole section/sibling group excluded by evidence budget.",
                                           group.citation.chunk_ids))
            continue
        included.append(group)
        rendered, token_count = candidate_text, candidate_count
        if len(group.spans) > 1:
            warnings.append(ContextWarning("source_gap", "Disconnected source excerpts; intervening content was not supplied.",
                                           group.citation.chunk_ids))
    return ContextPackage(tuple(included), tuple(warnings), mode, reference_date, rendered,
                          token_count, max_tokens, counter.name)
