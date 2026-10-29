"""Reusable Q&A orchestration. Imports perform no credential, model or network IO."""
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from typing import Iterator, Literal

from northstar_atlas.embeddings import EmbeddingProvider
from northstar_atlas.evidence import CitationReference
from northstar_atlas.generation import (
    GenerationError, GroundedAnswer, PreparedAnswer, PROMPT_VERSION, SCHEMA_VERSION, VALIDATION_VERSION,
    generate_answer, prepare_answer, render_answer,
)
from northstar_atlas.grounding import (AbstentionRequirements, ConditionCoverage, EvidenceObligation,
                                      ReviewedExtractiveScope)
from northstar_atlas.ingestion.document_loader import RetrievalMode
from northstar_atlas.llm import (
    GenerationLimits, GenerationRequest, GenerationResponse, LLMProvider,
    MODEL_ID, QwenTokenBound, TokenUsage,
)
from northstar_atlas.vector_store.contracts import RetrievalStore
from northstar_atlas.project import validate_project_root

ErrorCode = Literal["invalid_input", "configuration_failed", "retrieval_failed",
                    "context_failed", "provider_failed", "response_rejected", "internal_error"]
Stage = Literal["input", "configuration", "retrieval", "context", "provider", "validation", "complete"]

_MESSAGES: dict[ErrorCode, str] = {
    "invalid_input": "Please provide a valid question and retrieval options.",
    "configuration_failed": "The question-answering service is not configured for this request.",
    "retrieval_failed": "Policy evidence could not be retrieved. No answer was generated.",
    "context_failed": "Complete, eligible evidence could not be prepared within the configured limits.",
    "provider_failed": "The answer provider could not complete the request. No answer was accepted.",
    "response_rejected": "The generated response did not pass validation and has been withheld.",
    "internal_error": "The request could not be completed. No answer was accepted.",
}


@dataclass(frozen=True)
class PublicStatement:
    """Exact accepted wording and its source associations, never a UI summary."""
    text: str
    citations: tuple[CitationReference, ...]
    origin: Literal["model", "application_reviewed_source"] = "model"


@dataclass(frozen=True)
class PublicQuotation:
    """Original accepted quotation, not the whitespace-normalized source match."""
    text: str
    citation: CitationReference
    match_kind: Literal["exact", "whitespace_normalized"]


@dataclass(frozen=True)
class PublicPresentation:
    statements: tuple[PublicStatement, ...] = ()
    quotations: tuple[PublicQuotation, ...] = ()
    clarification_question: str | None = None
    missing_information: tuple[str, ...] = ()
    abstention_basis: Literal["documented_gap", "insufficient_evidence"] | None = None


@dataclass(frozen=True)
class PublicAnswer:
    kind: Literal["answer", "clarify", "abstain"]
    text: str
    citations: tuple[CitationReference, ...]
    coverage: ConditionCoverage
    presentation: PublicPresentation | None = None


def _accepted_presentation(answer: GroundedAnswer) -> PublicPresentation:
    """Project only after generate_answer AND the legacy render guard succeed.

    Keep this separate from validation: copying does not prove semantic support.
    No rendered-string parsing, provider bodies or operational diagnostics enter
    this public shape. Resolved obligations remain primary answer statements.
    """
    resolved = tuple(a for a in answer.obligation_answers if a.statement is not None)
    statements = tuple(PublicStatement(c.text, tuple(dict.fromkeys(s.citation for s in c.support)))
                       for c in answer.claims) + tuple(
        PublicStatement(a.statement, tuple(dict.fromkeys(s.citation for s in a.support)), a.statement_origin)
        for a in resolved)
    supports = tuple(s for claim in answer.claims for s in claim.support) + tuple(
        s for entry in resolved for s in entry.support)
    quotations = tuple(dict.fromkeys(PublicQuotation(s.quote, s.citation, s.match.kind) for s in supports))
    basis = None
    if answer.kind == "abstain":
        basis = answer.abstention.basis if answer.abstention is not None else "insufficient_evidence"
    return PublicPresentation(statements, quotations, answer.clarification_question,
                              answer.missing_information, basis)


@dataclass(frozen=True)
class PublicError:
    code: ErrorCode
    message: str


@dataclass(frozen=True)
class EvidenceTrace:
    """Developer provenance without source text, prompts or provider bodies."""
    citation: CitationReference
    chunk_id: str
    source_path: str
    source_spans: tuple[tuple[int, int, str], ...]
    required_context_ids: tuple[str, ...]


@dataclass(frozen=True)
class QADiagnostics:
    """Separate from the public projection; do not display wholesale to users."""
    stage: Stage
    backend: str
    mode: RetrievalMode | None
    reference_date: date | None
    corpus_fingerprint: str
    model: str
    prompt_version: str = PROMPT_VERSION
    schema_version: str = SCHEMA_VERSION
    validation_version: str = VALIDATION_VERSION
    evidence: tuple[EvidenceTrace, ...] = ()
    warning_codes: tuple[str, ...] = ()
    input_token_bound: int | None = None
    output_token_limit: int | None = None
    context_token_limit: int | None = None
    provider_calls: int = 0
    usage: TokenUsage | None = None
    quotation_matches: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    application_owned_ids: tuple[str, ...] = ()
    retrieval_strategy: str = "dense"


@dataclass(frozen=True)
class QASuccess:
    public: PublicAnswer
    diagnostics: QADiagnostics
    accepted: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True)
class QAFailure:
    public: PublicError
    diagnostics: QADiagnostics
    accepted: Literal[False] = field(default=False, init=False)


def _public_answer(answer: GroundedAnswer) -> PublicAnswer:
    """Shared publication projection, called only after existing validation."""
    rendered = render_answer(answer)
    supports = tuple(s for claim in answer.claims for s in claim.support) + tuple(
        s for entry in answer.obligation_answers for s in entry.support)
    return PublicAnswer(answer.kind, rendered, tuple(dict.fromkeys(s.citation for s in supports)),
                        answer.coverage, _accepted_presentation(answer))


def _prepared_diagnostics(diagnostics: QADiagnostics, prepared: PreparedAnswer) -> QADiagnostics:
    traces = tuple(EvidenceTrace(group.citation, source.chunk.chunk_id, str(source.chunk.source_path),
                                 tuple((s.source_start, s.source_end, s.role) for s in source.segments),
                                 source.required_context_ids)
                   for group in prepared.context.groups for source in group.sources)
    return replace(diagnostics, evidence=traces, prompt_version=prepared.prompt_version,
                   schema_version=prepared.schema_version, validation_version=prepared.validation_version,
                   warning_codes=tuple(w.code for w in prepared.context.warnings),
                   input_token_bound=prepared.request.input_token_bound,
                   output_token_limit=prepared.request.output_tokens,
                   context_token_limit=prepared.request.context_tokens)


def _accepted_diagnostics(diagnostics: QADiagnostics, answer: GroundedAnswer) -> QADiagnostics:
    supports = tuple(s for claim in answer.claims for s in claim.support) + tuple(
        s for entry in answer.obligation_answers for s in entry.support)
    return replace(diagnostics, stage="complete", quotation_matches=tuple(s.match.kind for s in supports),
                   application_owned_ids=tuple(a.obligation_id for a in answer.obligation_answers
                                               if a.statement_origin == "application_reviewed_source"))


class QAConfigurationError(ValueError):
    """Safe setup failure, distinct from a per-question policy abstention."""
    def __init__(self):
        self.public = PublicError("configuration_failed", _MESSAGES["configuration_failed"])
        super().__init__(self.public.message)


class _ProviderFailure(Exception):
    pass


class _ObservedProvider:
    """Attribute transport failures without interpreting provider exception strings."""
    def __init__(self, provider: LLMProvider, *, max_calls: int | None = None):
        self.provider = provider
        self.max_calls = max_calls
        self.calls = 0
        self.usage: TokenUsage | None = None

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise GenerationError("inference allowance exhausted")
        self.calls += 1
        try:
            response = self.provider.generate(request)
        except Exception:
            raise _ProviderFailure() from None
        if not isinstance(response, GenerationResponse):
            raise GenerationError("invalid provider response type")
        if self.max_calls is not None and (type(response.attempts) is not int or response.attempts != 1):
            raise GenerationError("provider did not use the required single attempt")
        usage = response.usage
        if isinstance(usage, TokenUsage) and all(type(n) is int and n >= 0 for n in
                                               (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)):
            self.usage = TokenUsage(usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)
        return response


class QAService:
    """Reuse initialized adapters across sequential questions; no retry or fallback.

    Adapters and reviewed requirements are trusted caller inputs. Validators bind
    requirements to the evidence actually returned for each question. The service
    never derives requirements from question wording or evaluation answer keys.
    """
    def __init__(self, store: RetrievalStore, embedder: EmbeddingProvider | None, provider: LLMProvider, *,
                 corpus_fingerprint: str, backend: str = "custom", model: str = MODEL_ID,
                 strategy: str = "dense", budget=None):
        try:
            QwenTokenBound(model)
            if (backend not in ("chroma", "pinecone", "custom") or strategy not in ("dense", "bm25", "hybrid") or
                    not isinstance(corpus_fingerprint, str) or not corpus_fingerprint.strip() or
                    not callable(getattr(store, "search", None)) or
                    not callable(getattr(provider, "generate", None))):
                raise ValueError
        except (TypeError, ValueError):
            raise QAConfigurationError() from None
        self._store, self._embedder, self._provider = store, embedder, provider
        self._fingerprint, self._backend, self._model = corpus_fingerprint, backend, model
        self._strategy = strategy
        self._budget = budget

    def ask(self, question: str, *, mode: RetrievalMode = "approved_only", top_k: int = 5,
            reference_date: date | None = None, limits: GenerationLimits = GenerationLimits(),
            obligations: tuple[EvidenceObligation, ...] = (),
            abstention: AbstentionRequirements | None = None,
            extractive_scope: ReviewedExtractiveScope | None = None,
            reservation=None, invocation_id: str | None = None) -> QASuccess | QAFailure:
        started = perf_counter()
        as_of = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
        if self._budget is not None and reservation is None:
            from .live_budget import WorkflowRequest, LINEAR, BudgetError
            try:
                with self._budget.reserve(WorkflowRequest(invocation_id, LINEAR, question, mode, self._backend, as_of)) as permit:
                    result = self.ask(question, mode=mode, top_k=top_k, reference_date=as_of, limits=limits,
                                      obligations=obligations, abstention=abstention,
                                      extractive_scope=extractive_scope, reservation=permit)
                return result
            except (BudgetError, OSError):
                return QAFailure(PublicError('configuration_failed', _MESSAGES['configuration_failed']),
                                 QADiagnostics('configuration', self._backend, None, None, self._fingerprint, self._model))
        observed = _ObservedProvider(self._provider)
        diagnostics = QADiagnostics("input", self._backend, None, None, self._fingerprint, self._model,
                                    retrieval_strategy=self._strategy)

        def failure(code: ErrorCode, stage: Stage) -> QAFailure:
            return QAFailure(PublicError(code, _MESSAGES[code]), replace(
                diagnostics, stage=stage, provider_calls=observed.calls, usage=observed.usage,
                elapsed_seconds=perf_counter() - started))

        if (not isinstance(question, str) or not question.strip() or len(question) > 16000 or
                mode not in ("approved_only", "draft_review") or type(top_k) is not int or not 1 <= top_k <= 100 or
                (reference_date is not None and type(reference_date) is not date) or
                not isinstance(limits, GenerationLimits)):
            return failure("invalid_input", "input")
        as_of = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
        diagnostics = replace(diagnostics, mode=mode, reference_date=as_of)

        if reservation is not None:
            from .live_budget import WorkflowReservation, StageProvider, LINEAR
            try:
                if not isinstance(reservation, WorkflowReservation):
                    raise ValueError
                if self._budget is not None:
                    reservation.check_coordinator(self._budget)
                reservation.check_request(profile=LINEAR, question=question, mode=mode,
                                          backend=self._backend, reference_date=as_of)
                observed = _ObservedProvider(StageProvider(reservation, self._provider), max_calls=1)
            except ValueError:
                return failure('configuration_failed', 'configuration')

        try:
            hits = self._store.search(question, self._embedder, corpus_fingerprint=self._fingerprint,
                                      mode=mode, top_k=top_k, reference_date=as_of)
        except Exception:
            return failure("retrieval_failed", "retrieval")
        try:
            prepared = prepare_answer(question, hits, mode=mode, reference_date=as_of, limits=limits,
                                      model=self._model, corpus_fingerprint=self._fingerprint,
                                      obligations=obligations, abstention=abstention,
                                      extractive_scope=extractive_scope)
            diagnostics = _prepared_diagnostics(diagnostics, prepared)
        except GenerationError:
            return failure("context_failed", "context")
        except Exception:
            return failure("internal_error", "context")
        try:
            result = generate_answer(prepared, observed)
            public = _public_answer(result.answer)
            return QASuccess(public, replace(_accepted_diagnostics(diagnostics, result.answer),
                                            provider_calls=observed.calls, usage=observed.usage,
                                            elapsed_seconds=perf_counter() - started))
        except _ProviderFailure:
            return failure("provider_failed", "provider")
        except GenerationError:
            return failure("response_rejected", "validation")
        except Exception:
            return failure("internal_error", "validation")


@contextmanager
def open_retriever(root: Path, backend: str, *, strategy: str = "dense", snapshot_id: str | None = None):
    """Verified retrieval; an explicit snapshot pins Pinecone's active generation.

    Never index/provision, switch publication, or regenerate document vectors.
    """
    root = validate_project_root(root)
    if strategy not in ("dense", "bm25", "hybrid") or backend not in ("chroma", "pinecone"):
        raise GenerationError("unsupported retrieval backend or strategy")
    if strategy != "dense":
        from northstar_atlas.vector_store.catalog import load_catalog
        from northstar_atlas.vector_store.bm25 import BM25Store
        catalog = load_catalog(root, snapshot_id)
        lexical = BM25Store(catalog)
        if strategy == "bm25":
            yield lexical, None, catalog.fingerprint
        else:
            from northstar_atlas.vector_store.hybrid import HybridStore
            options = {} if snapshot_id is None else {"snapshot_id": snapshot_id}
            with open_retriever(root, backend, **options) as (dense, embedder, fingerprint):
                if fingerprint != catalog.fingerprint or embedder.model != catalog.config.model:
                    raise GenerationError("hybrid corpus/model configuration differs")
                yield HybridStore(dense, lexical), embedder, fingerprint
        return
    from northstar_atlas.local_embeddings import MiniLMProvider
    from northstar_atlas.vector_store import ChromaStore, IndexConfig, corpus_snapshot
    _, fingerprint = corpus_snapshot(root)
    embedder = MiniLMProvider(root)
    if backend == "chroma":
        yield ChromaStore(root, IndexConfig(embedder.model, embedder.max_input_tokens)), embedder, fingerprint
    elif backend == "pinecone":
        from northstar_atlas.vector_store.pinecone_cloud import check_confirmation, load_settings as pinecone_settings, sdk_client
        from northstar_atlas.vector_store.pinecone_workflow import RequestMeter, active_snapshot, connect_target, open_store
        settings = pinecone_settings(root)
        check_confirmation(root, settings)
        meter = RequestMeter()
        with sdk_client(settings) as client:
            meter.attach(client)
            target, index = connect_target(client, meter, settings)
            with index:
                snapshot = active_snapshot(root, target)
                if snapshot_id is not None and snapshot.snapshot_id != snapshot_id:
                    raise GenerationError("active Pinecone snapshot differs from requested snapshot")
                store = open_store(root, snapshot, target, client, index)
                store.audit()
                yield store, embedder, fingerprint
    else:
        raise GenerationError("unsupported retrieval backend")


@contextmanager
def open_qa_service(root: Path, *, backend: str | None = None, provider: LLMProvider | None = None,
                    allow_paid: bool = False, model: str = MODEL_ID, strategy: str = "dense",
                    snapshot_id: str | None = None, budget=None, reservation=None) -> Iterator[QAService]:
    """Open reusable configured adapters; injected providers remain caller-owned.

    No inference occurs during setup. Configured Nebius requires explicit paid
    opt-in; selecting Pinecone opens the existing remote retrieval session. Offline
    callers should inject QAService adapters directly or mock the opener in tests.
    Configuration/cleanup failures are safe errors; caller-body errors propagate.
    """
    resources = ExitStack()
    try:
        root = validate_project_root(root)
        selected = os.environ.get("VECTOR_STORE", "chroma") if backend is None else backend
        if selected not in ("chroma", "pinecone") or strategy not in ("dense", "bm25", "hybrid"):
            raise QAConfigurationError()
        QwenTokenBound(model)
        if budget is None and os.environ.get('NORTHSTAR_CONTROLLED_LIVE') == '1':
            from .live_budget import configured_budget
            budget = configured_budget(root)
        if budget is not None:
            if reservation is None:
                budget.check_ready()
            else:
                reservation.check_coordinator(budget)
        if provider is not None and not callable(getattr(provider, "generate", None)):
            raise QAConfigurationError()
        if provider is None:
            if allow_paid is not True:
                raise QAConfigurationError()
            from northstar_atlas.nebius import NebiusProvider, load_settings
            settings = load_settings(root)
            if settings.model != model:
                raise QAConfigurationError()
            options = {'requires_budget': True} if budget is not None else {}
            provider = resources.enter_context(NebiusProvider(settings, allow_paid=True, max_attempts=1, **options))
        options = {} if strategy == "dense" else {"strategy": strategy, "snapshot_id": snapshot_id}
        store, embedder, fingerprint = resources.enter_context(open_retriever(root, selected, **options))
        service = QAService(store, embedder, provider, corpus_fingerprint=fingerprint, backend=selected,
                            model=model, strategy=strategy, budget=budget)
    except Exception:
        try:
            resources.close()
        except Exception:
            pass  # Do not expose an untrusted cleanup error over the safe setup error.
        raise QAConfigurationError() from None
    try:
        yield service
    except BaseException:
        try:
            resources.close()
        except Exception:
            pass  # Preserve the caller's original exception.
        raise
    else:
        try:
            resources.close()
        except Exception:
            raise QAConfigurationError() from None
