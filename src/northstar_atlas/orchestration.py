"""Opt-in, bounded local LangGraph workflow; no clients or models opened on import.

Availability assessment is not semantic completeness. Providers/catalogs are
trusted injected dependencies; callers cannot submit graph state or callbacks.
Only the publication node returns a public result. Defaults elsewhere stay linear.
"""
from contextlib import contextmanager, ExitStack
from contextvars import Context
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Iterator, Literal, Mapping, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langsmith import tracing_context

from northstar_atlas.context import build_context
from northstar_atlas.embeddings import EmbeddingProvider
from northstar_atlas.evidence import ContextPackage, validate_citations
from northstar_atlas.generation import (
    GenerationError, GroundedAnswer, PreparedAnswer, NO_EVIDENCE_TEXT,
    generate_answer, prepare_answer, validate_answer,
)
from northstar_atlas.grounding import (
    AbstentionRequirements, EvidenceObligation, ReviewedExtractiveScope,
    bind_extractive_obligations, validate_abstention, validate_obligations,
)
from northstar_atlas.ingestion.document_loader import RetrievalMode
from northstar_atlas.llm import GenerationLimits, LLMProvider, MODEL_ID, QwenTokenBound
from northstar_atlas.project import validate_project_root
from northstar_atlas.qa import (
    ErrorCode, Stage, PublicError, QADiagnostics, QAFailure, QASuccess, QAConfigurationError,
    _MESSAGES, _ObservedProvider, _ProviderFailure, _public_answer,
    _prepared_diagnostics, _accepted_diagnostics, open_retriever,
)
from northstar_atlas.vector_store.catalog import ChunkCatalog, load_catalog
from northstar_atlas.vector_store.contracts import RetrievalHit, RetrievalStore, SearchHit

Strategy = Literal['dense', 'bm25', 'hybrid']


@dataclass(frozen=True)
class CallerReviewedClarification:
    """Trusted, question/mode-scoped authored prompt; no policy guidance allowed."""
    question: str
    mode: RetrievalMode
    clarification_question: str
    missing_information: tuple[str, ...]


@dataclass(frozen=True)
class GraphRequest:
    question: str = field(repr=False)
    mode: RetrievalMode
    reference_date: date
    top_k: int
    limits: GenerationLimits
    obligations: tuple[EvidenceObligation, ...] = field(repr=False)
    abstention: AbstentionRequirements | None = field(repr=False)
    extractive_scope: ReviewedExtractiveScope | None = field(repr=False)
    clarification: CallerReviewedClarification | None = field(repr=False)


@dataclass(frozen=True)
class EvidenceAssessment:
    readiness: Literal['ready', 'insufficient', 'clarify', 'blocked']
    semantic_coverage: Literal['not_assessed', 'reviewed_inputs_bound', 'reviewed_inputs_missing']
    reason_codes: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttemptTrace:
    strategy: Strategy
    direct_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]
    warning_codes: tuple[str, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class GraphTrace:
    """Developer-only diagnostics: no question, source text, prompt or candidate."""
    nodes: tuple[str, ...] = ()
    retrieval_attempts: int = 0
    attempts: tuple[AttemptTrace, ...] = ()
    assessments: tuple[EvidenceAssessment, ...] = ()
    publication: Literal['passed', 'failed', 'not_run'] = 'not_run'


class GraphState(TypedDict, total=False):
    request: GraphRequest
    active_strategy: Strategy
    attempt_count: int
    retrieved_evidence: tuple[RetrievalHit, ...]
    assessment: EvidenceAssessment
    prepared: PreparedAnswer | None
    generation_result: GroundedAnswer | None
    validation_result: Literal['passed', 'failed', 'not_run']
    terminal_intent: Literal['clarify', 'abstain'] | None
    failure: tuple[ErrorCode, Stage] | None
    diagnostics: QADiagnostics
    trace: GraphTrace
    final: QASuccess | QAFailure


@dataclass
class _Invocation:
    provider: _ObservedProvider
    started: float
    # Retain only safe progress across framework exceptions, never candidates or
    # prompts. This object belongs to one invocation, not the reusable service.
    trace: GraphTrace = field(default_factory=GraphTrace)
    diagnostics: QADiagnostics | None = None


def _recording_node(method, *, with_runtime: bool):
    def run(state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        update = method(state, runtime) if with_runtime else method(state)
        runtime.context.trace = update.get('trace', state['trace'])
        runtime.context.diagnostics = update.get('diagnostics', state['diagnostics'])
        return update
    return run


class GraphQAService:
    """Reusable sequential service with 1–2 configured searches and <=1 inference.

    A verified catalog is required, including for dense-only plans: it separates
    absent reviewed sources from fabricated/stale requirements before retrieval.
    Injected providers must perform one transport attempt. The configured factory
    enforces this for Nebius; arbitrary caller-written provider internals are trusted.
    """
    def __init__(self, stores: Mapping[Strategy, RetrievalStore], embedder: EmbeddingProvider | None,
                 provider: LLMProvider, *, catalog: ChunkCatalog,
                 strategy_plan: tuple[Strategy, ...] = ('dense', 'bm25'),
                 backend: str = 'custom', model: str = MODEL_ID):
        try:
            QwenTokenBound(model)
            if (not isinstance(catalog, ChunkCatalog) or backend not in ('chroma', 'pinecone', 'custom')
                    or not isinstance(strategy_plan, tuple) or not 1 <= len(strategy_plan) <= 2
                    or len(set(strategy_plan)) != len(strategy_plan)
                    or any(s not in ('dense', 'bm25', 'hybrid') for s in strategy_plan)
                    or any(not callable(getattr(stores.get(s), 'search', None)) for s in strategy_plan)
                    or not callable(getattr(provider, 'generate', None))
                    or getattr(provider, '_attempts', 1) != 1):
                raise ValueError
            catalog.check_current(catalog.fingerprint)
            if embedder is not None and embedder.model != catalog.config.model:
                raise ValueError
        except Exception:
            raise QAConfigurationError() from None
        self._stores = MappingProxyType({s: stores[s] for s in strategy_plan})
        self._embedder, self._provider, self._catalog = embedder, provider, catalog
        self._plan, self._backend, self._model = strategy_plan, backend, model
        self.last_trace = GraphTrace()
        graph = StateGraph(GraphState, context_schema=_Invocation)
        for name in ('router', 'retriever', 'assessor', 'generator', 'validator'):
            graph.add_node(name, _recording_node(getattr(self, '_' + name),
                                                with_runtime=name in ('generator', 'validator')))
        graph.add_edge(START, 'router')
        graph.add_conditional_edges('router', lambda s: 'validator' if s.get('failure') or s.get('terminal_intent') else 'retriever')
        graph.add_conditional_edges('retriever', lambda s: 'validator' if s.get('failure') else 'assessor')
        graph.add_conditional_edges('assessor', self._after_assessment)
        graph.add_edge('generator', 'validator')
        graph.add_edge('validator', END)
        self._graph = graph.compile()  # No checkpointer, retries, cache or interrupts.

    def ask(self, question: str, *, mode: RetrievalMode = 'approved_only', top_k: int = 5,
            reference_date: date | None = None, limits: GenerationLimits = GenerationLimits(),
            obligations: tuple[EvidenceObligation, ...] = (),
            abstention: AbstentionRequirements | None = None,
            extractive_scope: ReviewedExtractiveScope | None = None,
            clarification: CallerReviewedClarification | None = None) -> QASuccess | QAFailure:
        started = perf_counter()
        as_of = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
        request = GraphRequest(question, mode, as_of, top_k, limits, obligations, abstention,
                               extractive_scope, clarification)
        invocation = _Invocation(_ObservedProvider(self._provider, max_calls=1), started)
        state: GraphState = dict(request=request, attempt_count=0, retrieved_evidence=(),
            prepared=None, generation_result=None, terminal_intent=None, failure=None, trace=GraphTrace(),
            validation_result='not_run', diagnostics=QADiagnostics(
                'input', self._backend, None, None, self._catalog.fingerprint, self._model,
                retrieval_strategy=self._plan[0]))
        self.last_trace = GraphTrace()

        def invoke():
            # Fresh contextvars also isolates inherited Runnable callbacks/config.
            with tracing_context(enabled=False, parent=False):
                return self._graph.invoke(state, context=invocation,
                                          config={'recursion_limit': 10, 'callbacks': []})
        try:
            result = Context().run(invoke)
            self.last_trace = result['trace']
            return result['final']
        except Exception:
            # Includes recursion/invariant failures: no partial output escapes.
            state['trace'] = invocation.trace
            state['attempt_count'] = invocation.trace.retrieval_attempts
            state['diagnostics'] = invocation.diagnostics or state['diagnostics']
            state['failure'] = ('internal_error', 'validation')
            result = self._publish(state, invocation)
            self.last_trace = result['trace']
            return result['final']

    @staticmethod
    def _visit(state: GraphState, node: str) -> GraphTrace:
        return replace(state['trace'], nodes=(*state['trace'].nodes, node))

    def _package(self, hits, request: GraphRequest) -> ContextPackage:
        return build_context(hits, counter=QwenTokenBound(self._model),
            max_tokens=request.limits.context_tokens-request.limits.output_tokens-request.limits.margin_tokens,
            mode=request.mode, reference_date=request.reference_date)

    def _reviewed_preflight(self, request: GraphRequest) -> None:
        """Bind against source catalog only; never inject this evidence into results."""
        if not isinstance(request.obligations, tuple):
            raise ValueError
        if request.abstention is not None:
            if request.obligations or request.extractive_scope is not None:
                raise ValueError
            required = request.abstention.limitations
        else:
            required = request.obligations
        if not required and request.extractive_scope is None:
            if request.abstention is not None:
                raise ValueError
            return
        ids = tuple(dict.fromkeys(key for o in required for key in o.citation.chunk_ids))
        hits = tuple(SearchHit(self._catalog.by_id[key], 0., self._catalog.config.model,
                              self._catalog.siblings(self._catalog.by_id[key], request.mode, request.reference_date))
                     for key in ids)
        for h in hits:
            self._catalog.validate_hit(h, request.mode, request.reference_date)
        # Source binding preflight is not a generation prompt. It must not depend
        # on the question's budget; actual retrieval gets the configured budget.
        package = self._package(hits, replace(request, limits=GenerationLimits(context_tokens=262144)))
        if any(w.code in ('source_gap', 'budget_excluded') for w in package.warnings):
            raise ValueError
        if request.abstention is not None:
            validate_abstention(request.abstention, package)
        else:
            validate_obligations(request.obligations, package)
        if request.extractive_scope is not None:
            bind_extractive_obligations(request.obligations, package, request.extractive_scope,
                                       question=request.question)

    def _router(self, state: GraphState) -> dict:
        r = state['request']
        update = dict(trace=self._visit(state, 'router'), active_strategy=self._plan[0])
        if (not isinstance(r.question, str) or not r.question.strip() or len(r.question) > 16000
                or r.mode not in ('approved_only', 'draft_review') or type(r.top_k) is not int
                or not 1 <= r.top_k <= 5 or type(r.reference_date) is not date
                or not isinstance(r.limits, GenerationLimits)):
            return {**update, 'failure': ('invalid_input', 'input')}
        update['diagnostics'] = replace(state['diagnostics'], mode=r.mode, reference_date=r.reference_date)
        try:
            self._catalog.check_current(self._catalog.fingerprint)
            hint = r.clarification
            if hint is not None:
                if (not isinstance(hint, CallerReviewedClarification) or hint.question != r.question
                        or hint.mode != r.mode or r.obligations or r.abstention or r.extractive_scope
                        or not isinstance(hint.missing_information, tuple)):
                    raise ValueError
                # Validate the authored template at the same publication boundary.
                self._terminal_answer(r, 'clarify')
                return {**update, 'terminal_intent': 'clarify'}
            self._reviewed_preflight(r)
        except Exception:
            return {**update, 'failure': ('configuration_failed', 'configuration')}
        return update

    def _retriever(self, state: GraphState) -> dict:
        trace = self._visit(state, 'retriever')
        count = state['attempt_count']
        if not 0 <= count < len(self._plan) <= 2:
            return dict(trace=trace, failure=('internal_error', 'retrieval'))
        strategy = self._plan[count]
        r = state['request']
        trace = replace(trace, retrieval_attempts=count+1)
        update = dict(trace=trace, attempt_count=count+1, active_strategy=strategy,
                      retrieved_evidence=(), prepared=None, generation_result=None,
                      diagnostics=replace(state['diagnostics'], retrieval_strategy=strategy,
                                          evidence=(), warning_codes=()))
        start = perf_counter()
        try:
            self._catalog.check_current(self._catalog.fingerprint)
            hits = tuple(self._stores[strategy].search(r.question, self._embedder,
                corpus_fingerprint=self._catalog.fingerprint, mode=r.mode, top_k=r.top_k,
                reference_date=r.reference_date))
        except Exception:
            return {**update, 'failure': ('retrieval_failed', 'retrieval')}
        try:
            if len(hits) > r.top_k or len({h.chunk.chunk_id for h in hits}) != len(hits):
                raise ValueError
            for h in hits:
                self._catalog.validate_hit(h, r.mode, r.reference_date)
        except Exception:
            return {**update, 'failure': ('context_failed', 'context')}
        attempt = AttemptTrace(strategy, tuple(h.chunk.chunk_id for h in hits), (), (), perf_counter()-start)
        return {**update, 'retrieved_evidence': hits,
                'trace': replace(trace, attempts=(*trace.attempts, attempt))}

    def _assessor(self, state: GraphState) -> dict:
        r = state['request']
        trace = self._visit(state, 'assessor')
        required = r.abstention.limitations if r.abstention is not None else r.obligations
        semantic = 'reviewed_inputs_bound' if required else 'not_assessed'
        try:
            package = self._package(state['retrieved_evidence'], r)
            codes = tuple(w.code for w in package.warnings)
            attempt = replace(trace.attempts[-1], citation_ids=tuple(g.citation.reference_id for g in package.groups),
                              warning_codes=codes)
            trace = replace(trace, attempts=(*trace.attempts[:-1], attempt))
            if any(code in ('budget_excluded', 'source_gap', 'unrelated_context', 'ineligible') for code in codes):
                raise GenerationError('incomplete context')
            available = {s.chunk.chunk_id for group in package.groups for s in group.sources}
            missing = tuple(o.obligation_id for o in required if not set(o.citation.chunk_ids) <= available)
            if missing or not package.groups:
                assessment = EvidenceAssessment('insufficient', 'reviewed_inputs_missing' if required else 'not_assessed',
                                                ('missing_reviewed_evidence' if required else 'empty_evidence',), missing)
                update = dict(assessment=assessment, trace=replace(trace, assessments=(*trace.assessments, assessment)))
                if state['attempt_count'] < len(self._plan):
                    return update
                if required:
                    return {**update, 'failure': ('context_failed', 'context')}
                return {**update, 'terminal_intent': 'abstain'}
            prepared = prepare_answer(r.question, state['retrieved_evidence'], mode=r.mode,
                reference_date=r.reference_date, limits=r.limits, model=self._model,
                corpus_fingerprint=self._catalog.fingerprint, obligations=r.obligations,
                abstention=r.abstention, extractive_scope=r.extractive_scope)
            if prepared.context != package:
                raise GenerationError('context changed during pure preparation')
            assessment = EvidenceAssessment('ready', semantic)
            return dict(prepared=prepared, assessment=assessment,
                        trace=replace(trace, assessments=(*trace.assessments, assessment)),
                        diagnostics=_prepared_diagnostics(state['diagnostics'], prepared))
        except Exception:
            assessment = EvidenceAssessment('blocked', semantic, ('context_invalid_or_over_budget',))
            return dict(failure=('context_failed', 'context'), assessment=assessment,
                        trace=replace(trace, assessments=(*trace.assessments, assessment)))

    @staticmethod
    def _after_assessment(state: GraphState) -> str:
        if state.get('failure') or state.get('terminal_intent'):
            return 'validator'
        return 'generator' if state['assessment'].readiness == 'ready' else 'retriever'

    def _generator(self, state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        update = dict(trace=self._visit(state, 'generator'))
        try:
            if state['assessment'].readiness != 'ready' or state['prepared'] is None:
                raise GenerationError('generation requires prepared evidence')
            result = generate_answer(state['prepared'], runtime.context.provider)
            if result.prepared is not state['prepared']:
                raise GenerationError('generation changed prepared evidence')
            return {**update, 'generation_result': result.answer}
        except _ProviderFailure:
            return {**update, 'failure': ('provider_failed', 'provider')}
        except GenerationError:
            return {**update, 'failure': ('response_rejected', 'validation')}
        except Exception:
            return {**update, 'failure': ('internal_error', 'validation')}

    def _validator(self, state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        return self._publish(state, runtime.context)

    def _terminal_answer(self, request: GraphRequest, kind: Literal['clarify', 'abstain']):
        hint = request.clarification
        body = dict(kind=kind, main_answer={'claims': [], 'obligations': {}},
            missing_information=list(hint.missing_information) if kind == 'clarify' else [NO_EVIDENCE_TEXT],
            clarification_question=hint.clarification_question if kind == 'clarify' else None,
            reason='missing_details' if kind == 'clarify' else 'insufficient_evidence')
        package = self._package((), request)
        return validate_answer(json.dumps(body), package), package

    def _publish(self, state: GraphState, invocation: _Invocation) -> dict:
        """The only creator of final results, including authored terminal replies."""
        trace = self._visit(state, 'validator')
        observed = invocation.provider
        diagnostics = replace(state['diagnostics'], provider_calls=observed.calls, usage=observed.usage,
                              elapsed_seconds=perf_counter()-invocation.started)
        failure = state.get('failure')
        if failure is None:
            try:
                r = state['request']
                if not 0 <= state['attempt_count'] <= 2 or observed.calls > 1:
                    raise GenerationError('invalid call counters')
                if state.get('terminal_intent'):
                    if r.obligations or r.abstention or r.extractive_scope or observed.calls:
                        raise GenerationError('terminal template cannot bypass requirements')
                    answer, package = self._terminal_answer(r, state['terminal_intent'])
                else:
                    prepared = state['prepared']
                    if (prepared is None or prepared.corpus_fingerprint != self._catalog.fingerprint
                            or prepared.context.mode != r.mode or prepared.context.reference_date != r.reference_date
                            or prepared.request.model != self._model or observed.calls != 1):
                        raise GenerationError('publication identity mismatch')
                    package, answer = prepared.context, state['generation_result']
                if not isinstance(answer, GroundedAnswer):
                    raise GenerationError('no validated answer')
                public = _public_answer(answer)
                validate_citations(public.citations, package)
                final = QASuccess(public, _accepted_diagnostics(diagnostics, answer))
                return dict(final=final, validation_result='passed', trace=replace(trace, publication='passed'))
            except Exception:
                failure = ('response_rejected', 'validation')
        code, stage = failure
        final = QAFailure(PublicError(code, _MESSAGES[code]), replace(diagnostics, stage=stage))
        return dict(final=final, validation_result='failed', trace=replace(trace, publication='failed'))


@contextmanager
def open_graph_qa_service(root: Path, *, snapshot_id: str, backend: str | None = None,
                          strategy_plan: tuple[Strategy, ...] = ('dense', 'bm25'),
                          provider: LLMProvider | None = None, allow_paid: bool = False,
                          model: str = MODEL_ID) -> Iterator[GraphQAService]:
    """Explicit opt-in. Reuse one dense adapter/model and local verified catalog.

    snapshot_id identifies an existing immutable local snapshot; never create or
    download one here. Selecting Pinecone retains its existing confirmation gate.
    Injected providers are caller-owned. Setup failures have safe error messages.
    """
    resources = ExitStack()
    try:
        root = validate_project_root(root)
        selected = os.environ.get('VECTOR_STORE', 'chroma') if backend is None else backend
        if (selected not in ('chroma', 'pinecone') or not isinstance(strategy_plan, tuple)
                or not 1 <= len(strategy_plan) <= 2 or len(set(strategy_plan)) != len(strategy_plan)
                or any(s not in ('dense', 'bm25', 'hybrid') for s in strategy_plan)):
            raise QAConfigurationError()
        QwenTokenBound(model)
        if provider is not None and (not callable(getattr(provider, 'generate', None))
                                     or getattr(provider, '_attempts', 1) != 1):
            raise QAConfigurationError()
        if provider is None and allow_paid is not True:
            raise QAConfigurationError()
        cat = load_catalog(root, snapshot_id)
        if provider is None:
            from northstar_atlas.nebius import NebiusProvider, load_settings
            settings = load_settings(root)
            if settings.model != model:
                raise QAConfigurationError()
            provider = resources.enter_context(NebiusProvider(settings, allow_paid=True, max_attempts=1))
        stores = {}
        embedder = None
        if any(s in ('dense', 'hybrid') for s in strategy_plan):
            dense, embedder, fingerprint = resources.enter_context(open_retriever(root, selected))
            if fingerprint != cat.fingerprint or embedder.model != cat.config.model:
                raise QAConfigurationError()
            stores['dense'] = dense
        if any(s in ('bm25', 'hybrid') for s in strategy_plan):
            from northstar_atlas.vector_store.bm25 import BM25Store
            stores['bm25'] = BM25Store(cat)
        if 'hybrid' in strategy_plan:
            from northstar_atlas.vector_store.hybrid import HybridStore
            stores['hybrid'] = HybridStore(stores['dense'], stores['bm25'])
        service = GraphQAService(stores, embedder, provider, catalog=cat,
                                 strategy_plan=strategy_plan, backend=selected, model=model)
    except BaseException as error:
        try:
            resources.close()
        except BaseException:
            pass
        if not isinstance(error, Exception):
            raise  # Cleanup also on interruption; preserve the original signal.
        raise QAConfigurationError() from None
    try:
        yield service
    except BaseException:
        try:
            resources.close()
        except Exception:
            pass
        raise
    else:
        try:
            resources.close()
        except Exception:
            raise QAConfigurationError() from None
