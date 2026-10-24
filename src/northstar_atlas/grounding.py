"""Literal quotation provenance and caller-reviewed evidence obligations; no IO."""
from dataclasses import dataclass, field
import re
from typing import Literal

from northstar_atlas.evidence import CitationReference, ContextPackage, EvidenceGroup, validate_citations


class GroundingError(ValueError):
    """Invalid quotation or obligation, with no model/source text in the error."""


@dataclass(frozen=True)
class QuotationMatch:
    kind: Literal["exact", "whitespace_normalized"]
    source_start: int
    source_end: int
    source_text: str = field(repr=False)


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs to one space; no case/punctuation/word normalization."""
    return " ".join(text.split())


def _normalized_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Map each normalized character back to its original end-exclusive interval."""
    characters: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    previous_end = 0
    for token in re.finditer(r"\S+", text):
        if characters:
            characters.append(" ")
            starts.append(previous_end)
            ends.append(token.start())
        characters.extend(token.group())
        starts.extend(range(token.start(), token.end()))
        ends.extend(range(token.start() + 1, token.end() + 1))
        previous_end = token.end()
    return "".join(characters), starts, ends


_NUMBER = re.compile(r"[+\-−]?(?:\d+(?:[.,:/-]\d+)*|[.,]\d+)(?:[eE][+\-−]?\d+)?")


def _bounded_find(text: str, quote: str, *, start: int = 0) -> int | None:
    """Reject word/number fragments, including signs, decimal parts and ranges.

    This is a conservative lexical guard, not a parser for numeric meaning or
    every locale's notation. Whitespace-separated qualifiers still need review.
    """
    def word(character: str) -> bool:
        return character.isalnum() or character == "_"
    numbers = tuple((match.start(), match.end()) for match in _NUMBER.finditer(text))
    cursor = start
    while (position := text.find(quote, cursor)) >= 0:
        end = position + len(quote)
        left = position == 0 or not (word(text[position - 1]) and word(quote[0]))
        right = end == len(text) or not (word(quote[-1]) and word(text[end]))
        whole_numbers = not any(start < position < stop or start < end < stop
                                for start, stop in numbers)
        if left and right and whole_numbers:
            return position
        cursor = position + 1
    return None


def match_quotation(quote: str, group: EvidenceGroup) -> QuotationMatch:
    """Match within ONE supplied contiguous span; preserve both originals.

    Exact matches take precedence. Otherwise only whitespace runs may differ.
    Interior deletions, changes, punctuation/case changes and reordered words
    cannot match. An excerpt can still omit a surrounding qualifier; this function
    does not determine relevance, authority, entailment or completeness.
    """
    if not isinstance(quote, str) or not quote.strip():
        raise GroundingError("quotation must contain non-whitespace text")
    for span in group.spans:
        start = _bounded_find(span.text, quote)
        if start is not None:
            end = start + len(quote)
            return QuotationMatch("exact", span.source_start + start, span.source_start + end, span.text[start:end])
    normalized_quote = normalize_whitespace(quote)
    for span in group.spans:
        normalized_source, starts, ends = _normalized_offsets(span.text)
        start = _bounded_find(normalized_source, normalized_quote)
        if start is not None:
            raw_start, raw_end = starts[start], ends[start + len(normalized_quote) - 1]
            return QuotationMatch("whitespace_normalized", span.source_start + raw_start,
                                  span.source_start + raw_end, span.text[raw_start:raw_end])
    raise GroundingError("quotation is not a contiguous supplied source match")


@dataclass(frozen=True)
class EvidenceObligation:
    """A trusted caller's reviewed source statement required in the main answer.

    It must bind to the current eligible citation snapshot. Callers must choose
    an explicit validation mode; no silent default changes older strict callers.
    Evidence-linked prose is not subjected to extractive equality or certified
    as semantically supported. Neither mode is an automatic semantic judgement.
    Policies/questions are never selected by document ID inside this module.
    """
    obligation_id: str
    citation: CitationReference
    source_quote: str
    validation_mode: Literal["extractive", "evidence_linked"]


@dataclass(frozen=True)
class ReviewedExtractiveSpan:
    """Caller-reviewed character offsets, end-exclusive, in the cited source."""
    obligation_id: str
    source_start: int
    source_end: int


@dataclass(frozen=True)
class ReviewedExtractiveScope:
    """Explicit opt-in for a reviewed question/mode; never inferred from model IDs."""
    question: str
    mode: Literal["approved_only", "draft_review"]
    spans: tuple[ReviewedExtractiveSpan, ...]


@dataclass(frozen=True)
class BoundExtractiveObligation:
    obligation: EvidenceObligation
    match: QuotationMatch


def bind_extractive_obligations(obligations: tuple[EvidenceObligation, ...],
                               package: ContextPackage, scope: ReviewedExtractiveScope, *,
                               question: str) -> tuple[BoundExtractiveObligation, ...]:
    """Verify caller spans against a trusted build_context snapshot, not semantics.

    Neither a caller-supplied offset nor an ID authenticates evidence. Retain the
    exact matched source only after citation, text, unique-interval and scope checks.
    Legacy first-match quotation behavior remains unchanged.
    """
    checked = validate_obligations(obligations, package)
    extracts = tuple(o for o in checked if o.validation_mode == "extractive")
    if (not isinstance(scope, ReviewedExtractiveScope) or not extracts
            or not isinstance(question, str) or not question.strip() or len(question) > 16000
            or scope.question != question or scope.mode not in ("approved_only", "draft_review")
            or scope.mode != package.mode or not isinstance(scope.spans, tuple)
            or any(not isinstance(s, ReviewedExtractiveSpan) for s in scope.spans)
            or tuple(s.obligation_id for s in scope.spans) != tuple(o.obligation_id for o in extracts)
            or any(w.code in ("budget_excluded", "source_gap") for w in package.warnings)):
        raise GroundingError("invalid reviewed extractive scope or required spans")
    groups = {group.citation.reference_id: group for group in package.groups}
    bound = []
    for obligation, reviewed in zip(extracts, scope.spans, strict=True):
        if (type(reviewed.source_start) is not int or type(reviewed.source_end) is not int
                or not 0 <= reviewed.source_start < reviewed.source_end):
            raise GroundingError("invalid reviewed source interval")
        group = groups[obligation.citation.reference_id]
        match = match_quotation(obligation.source_quote, group)
        intervals = set()
        normalized_quote = normalize_whitespace(obligation.source_quote)
        for span in group.spans:
            normalized, starts, ends = _normalized_offsets(span.text)
            cursor = 0
            while (position := _bounded_find(normalized, normalized_quote, start=cursor)) is not None:
                intervals.add((span.source_start + starts[position],
                               span.source_start + ends[position + len(normalized_quote) - 1]))
                if len(intervals) > 1:
                    raise GroundingError("reviewed quotation identifies ambiguous source intervals")
                cursor = position + 1
        if (match.source_start, match.source_end) != (reviewed.source_start, reviewed.source_end):
            raise GroundingError("reviewed source interval does not match supplied evidence")
        bound.append(BoundExtractiveObligation(obligation, match))
    return tuple(bound)


@dataclass(frozen=True)
class AbstentionRequirements:
    """Trusted caller's reviewed limitations, not a model's absence inference.

    Missing-information labels and basis require caller review. Source membership
    verifies provenance, not that the requested information is absent everywhere.
    """
    limitations: tuple[EvidenceObligation, ...]
    missing_information: tuple[str, ...]
    basis: Literal["documented_gap", "insufficient_evidence"]


@dataclass(frozen=True)
class ConditionCoverage:
    """Application-computed checks, not semantic completion.

    satisfied_ids passed their explicit structural/extractive validation mode;
    evidence_linked_ids still require semantic review.
    """
    status: Literal["not_assessed", "required_entries_present", "missing_required_statements"] = "not_assessed"
    satisfied_ids: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()
    extractive_ids: tuple[str, ...] = ()
    evidence_linked_ids: tuple[str, ...] = ()


def validate_obligations(obligations: tuple[EvidenceObligation, ...],
                         package: ContextPackage) -> tuple[EvidenceObligation, ...]:
    if not isinstance(obligations, tuple) or len(obligations) > 12:
        raise GroundingError("obligations must be a tuple with at most twelve reviewed statements")
    seen: set[str] = set()
    groups = {group.citation.reference_id: group for group in package.groups}
    for obligation in obligations:
        if (not isinstance(obligation, EvidenceObligation)
                or not isinstance(obligation.obligation_id, str)
                or not isinstance(obligation.citation, CitationReference)
                or obligation.validation_mode not in ("extractive", "evidence_linked")
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", obligation.obligation_id)
                or obligation.obligation_id in seen
                or not isinstance(obligation.source_quote, str) or len(obligation.source_quote) > 4000):
            raise GroundingError("invalid or duplicate evidence obligation")
        seen.add(obligation.obligation_id)
        try:
            citation = validate_citations([obligation.citation], package)[0]
        except ValueError:
            raise GroundingError("obligation citation is not supplied and eligible") from None
        match_quotation(obligation.source_quote, groups[citation.reference_id])
    return obligations


def validate_abstention(requirements: AbstentionRequirements,
                        package: ContextPackage) -> AbstentionRequirements:
    """Bind reviewed limitations to the current eligible source snapshot."""
    if (not isinstance(requirements, AbstentionRequirements)
            or requirements.basis not in ("documented_gap", "insufficient_evidence")
            or not requirements.limitations
            or not isinstance(requirements.missing_information, tuple)
            or not 1 <= len(requirements.missing_information) <= 12):
        raise GroundingError("invalid reviewed abstention requirements")
    labels = requirements.missing_information
    if (any(not isinstance(label, str) or not label.strip() or len(label) > 4000 for label in labels)
            or len(set(labels)) != len(labels)):
        raise GroundingError("invalid reviewed missing-information labels")
    validate_obligations(requirements.limitations, package)
    return requirements
