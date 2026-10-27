"""Opt-in, bounded local LangGraph workflow; no clients or models opened on import.

Availability assessment is not semantic completeness. Providers/catalogs are
trusted injected dependencies; callers cannot submit graph state or callbacks.
Only the publication node returns a public result. Defaults elsewhere stay linear.
"""
from contextlib import contextmanager, ExitStack
from contextvars import Context
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from inspect import Parameter, signature
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
    generate_answer, prepare_answer, render_answer, validate_answer,
)
from northstar_atlas.grounding import (
    AbstentionRequirements, EvidenceObligation, ReviewedExtractiveScope,
    bind_extractive_obligations, validate_abstention, validate_obligations,
)
from northstar_atlas.ingestion.document_loader import RetrievalMode
from northstar_atlas.llm import GenerationLimits, LLMProvider, MODEL_ID, QwenTokenBound
from northstar_atlas.project import validate_project_root
from northstar_atlas.orchestration_assessment import (
    PROFILE, SEMANTIC_PROFILE, PhaseALimits, PhaseAInvocation, PhaseBInvocation,
    GenerationStage, AssessmentCall, SemanticAssessment, SemanticContextError,
    RouteDecision, GradingAssessment, prepare_route, parse_route,
    prepare_grading, parse_grading, select_graded_hits, build_semantic_inventory,
    prepare_semantic_check, parse_semantic_assessment, prepare_regeneration_request,
    SemanticIssue,
)
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
    department_scope: tuple[str, ...] | None = None


@dataclass(frozen=True)
class GraphTrace:
    """Developer-only diagnostics: no question, source text, prompt or candidate."""
    nodes: tuple[str, ...] = ()
    retrieval_attempts: int = 0
    attempts: tuple[AttemptTrace, ...] = ()
    assessments: tuple[EvidenceAssessment, ...] = ()
    publication: Literal['passed', 'failed', 'not_run'] = 'not_run'
    route: RouteDecision | None = None
    gradings: tuple[GradingAssessment, ...] = ()
    calls: tuple[AssessmentCall, ...] = ()
    semantic_outcomes: tuple[str, ...] = ()
    usage_unknown: bool = False


class GraphState(TypedDict, total=False):
    request: GraphRequest
    active_strategy: Strategy
    route_decision: RouteDecision | None
    active_scope: tuple[str, ...] | None
    selected_evidence: tuple[RetrievalHit, ...]
    grading: GradingAssessment | None
    attempt_count: int
    retrieved_evidence: tuple[RetrievalHit, ...]
    assessment: EvidenceAssessment
    prepared: PreparedAnswer | None
    generation_result: GroundedAnswer | None
    validation_result: Literal['passed', 'failed', 'not_run']
    generation_attempt_count: int
    semantic_check_count: int
    semantic_assessment: SemanticAssessment | None
    regeneration_issue_codes: tuple[SemanticIssue, ...]
    terminal_intent: Literal['clarify', 'abstain'] | None
    failure: tuple[ErrorCode, Stage] | None
    diagnostics: QADiagnostics
    trace: GraphTrace
    final: QASuccess | QAFailure


@dataclass
class _Invocation:
    provider: _ObservedProvider | GenerationStage
    started: float
    workflow: PhaseAInvocation | PhaseBInvocation | None = None
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
    """Reusable sequential service with explicit, versioned graph profiles.

    bounded_v1 is unchanged. Phase A allows at most four calls. The Phase B
    semantic profile allows at most seven, and still requires an injected provider.

    A verified catalog is required, including for dense-only plans: it separates
    absent reviewed sources from fabricated/stale requirements before retrieval.
    Injected providers must perform one transport attempt. The configured factory
    enforces this for Nebius; arbitrary caller-written provider internals are trusted.
    """
    def __init__(self, stores: Mapping[Strategy, RetrievalStore], embedder: EmbeddingProvider | None,
                 provider: LLMProvider, *, catalog: ChunkCatalog,
                 strategy_plan: tuple[Strategy, ...] = ('dense', 'bm25'),
                 backend: str = 'custom', model: str = MODEL_ID,
                 profile: str = 'bounded_v1', phase_limits: PhaseALimits = PhaseALimits(), budget=None):
        try:
            QwenTokenBound(model)
            if (profile not in ('bounded_v1', PROFILE, SEMANTIC_PROFILE)
                    or not isinstance(phase_limits, PhaseALimits)
                    or not isinstance(catalog, ChunkCatalog) or backend not in ('chroma', 'pinecone', 'custom')
                    or not isinstance(strategy_plan, tuple) or not 1 <= len(strategy_plan) <= 2
                    or len(set(strategy_plan)) != len(strategy_plan)
                    or any(s not in ('dense', 'bm25', 'hybrid') for s in strategy_plan)
                    or any(not callable(getattr(stores.get(s), 'search', None)) for s in strategy_plan)
                    or not callable(getattr(provider, 'generate', None))
                    or getattr(provider, '_attempts', 1) != 1):
                raise ValueError
            if profile in (PROFILE, SEMANTIC_PROFILE):
                for strategy in strategy_plan:
                    parameter = signature(stores[strategy].search).parameters.get('department_scope')
                    if parameter is None or parameter.kind not in (Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD):
                        raise ValueError
            catalog.check_current(catalog.fingerprint)
            if embedder is not None and embedder.model != catalog.config.model:
                raise ValueError
        except Exception:
            raise QAConfigurationError() from None
        self._stores = MappingProxyType({s: stores[s] for s in strategy_plan})
        self._embedder, self._provider, self._catalog = embedder, provider, catalog
        self._plan, self._backend, self._model = strategy_plan, backend, model
        self._profile, self._phase_limits = profile, phase_limits
        self._budget = budget
        self.last_trace = GraphTrace()
        graph = StateGraph(GraphState, context_schema=_Invocation)
        for name in ('router', 'retriever', 'assessor', 'generator', 'semantic_checker', 'validator'):
            graph.add_node(name, _recording_node(getattr(self, '_' + name),
                                                with_runtime=name in ('router', 'assessor', 'generator',
                                                                      'semantic_checker', 'validator')))
        graph.add_edge(START, 'router')
        graph.add_conditional_edges('router', lambda s: 'validator' if s.get('failure') or s.get('terminal_intent') else 'retriever')
        graph.add_conditional_edges('retriever', lambda s: 'validator' if s.get('failure') else 'assessor')
        graph.add_conditional_edges('assessor', self._after_assessment)
        graph.add_conditional_edges('generator', self._after_generation)
        graph.add_conditional_edges('semantic_checker', self._after_semantic)
        graph.add_edge('validator', END)
        self._graph = graph.compile()  # No checkpointer, retries, cache or interrupts.

    def ask(self, question: str, *, mode: RetrievalMode = 'approved_only', top_k: int = 5,
            reference_date: date | None = None, limits: GenerationLimits = GenerationLimits(),
            obligations: tuple[EvidenceObligation, ...] = (),
            abstention: AbstentionRequirements | None = None,
            extractive_scope: ReviewedExtractiveScope | None = None,
            clarification: CallerReviewedClarification | None = None,
            reservation=None, invocation_id: str | None = None) -> QASuccess | QAFailure:
        started = perf_counter()
        as_of = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
        if self._budget is not None and reservation is None:
            from .live_budget import WorkflowRequest, GRAPH, BudgetError
            try:
                if self._profile != SEMANTIC_PROFILE:
                    raise BudgetError('unsupported controlled profile')
                with self._budget.reserve(WorkflowRequest(invocation_id, GRAPH, question, mode, self._backend, as_of)) as permit:
                    result = self.ask(question, mode=mode, top_k=top_k, reference_date=as_of, limits=limits,
                        obligations=obligations, abstention=abstention, extractive_scope=extractive_scope,
                        clarification=clarification, reservation=permit)
                return result
            except (BudgetError, OSError):
                return QAFailure(PublicError('configuration_failed', _MESSAGES['configuration_failed']),
                    QADiagnostics('configuration', self._backend, None, None, self._catalog.fingerprint, self._model))
        if reservation is not None:
            from .live_budget import WorkflowReservation, GRAPH
            try:
                if not isinstance(reservation, WorkflowReservation) or self._profile != SEMANTIC_PROFILE:
                    raise ValueError
                if self._budget is not None:
                    reservation.check_coordinator(self._budget)
                reservation.check_request(profile=GRAPH, question=question, mode=mode,
                                          backend=self._backend, reference_date=as_of)
            except ValueError:
                return QAFailure(PublicError('configuration_failed', _MESSAGES['configuration_failed']),
                    QADiagnostics('configuration', self._backend, None, None, self._catalog.fingerprint, self._model))
        request = GraphRequest(question, mode, as_of, top_k, limits, obligations, abstention,
                               extractive_scope, clarification)
        workflow = (PhaseAInvocation(self._provider, model=self._model, limits=self._phase_limits)
                    if self._profile == PROFILE else
                    PhaseBInvocation(self._provider, model=self._model, limits=self._phase_limits, reservation=reservation)
                    if self._profile == SEMANTIC_PROFILE else None)
        invocation = _Invocation(workflow.generation if workflow else _ObservedProvider(self._provider, max_calls=1),
                                 started, workflow)
        state: GraphState = dict(request=request, attempt_count=0, retrieved_evidence=(),
            route_decision=None, active_scope=None, selected_evidence=(), grading=None,
            prepared=None, generation_result=None, terminal_intent=None, failure=None, trace=GraphTrace(),
            validation_result='not_run', generation_attempt_count=0, semantic_check_count=0,
            semantic_assessment=None, regeneration_issue_codes=(), diagnostics=QADiagnostics(
                'input', self._backend, None, None, self._catalog.fingerprint, self._model,
                retrieval_strategy=self._plan[0]))
        self.last_trace = GraphTrace()

        def invoke():
            # Fresh contextvars also isolates inherited Runnable callbacks/config.
            with tracing_context(enabled=False, parent=False):
                return self._graph.invoke(state, context=invocation,
                                          config={'recursion_limit': 12 if self._profile == SEMANTIC_PROFILE else 10,
                                                  'callbacks': []})
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

    def _router(self, state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        r = state['request']
        update = dict(trace=self._visit(state, 'router'), active_strategy=self._plan[0])
        if (not isinstance(r.question, str) or not r.question.strip() or len(r.question) > 16000
                or r.mode not in ('approved_only', 'draft_review') or type(r.top_k) is not int
                or not 1 <= r.top_k <= 5 or type(r.reference_date) is not date
                or not isinstance(r.limits, GenerationLimits)):
            return {**update, 'failure': ('invalid_input', 'input')}
        update['diagnostics'] = replace(state['diagnostics'], mode=r.mode, reference_date=r.reference_date)
        try:
            if self._profile in (PROFILE, SEMANTIC_PROFILE) and (
                    r.limits.context_tokens > 32768 or r.limits.output_tokens > 2048
                    or r.limits.margin_tokens < 512):
                raise ValueError
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
        if runtime.context.workflow is not None:
            try:
                request = prepare_route(r.question, model=self._model, limits=self._phase_limits)
            except GenerationError:
                return {**update, 'failure': ('context_failed', 'context')}
            try:
                response = runtime.context.workflow.request('router', request)
                route = parse_route(response.text)
                return {**update, 'route_decision': route, 'active_scope': route.department_scope,
                        'trace': replace(update['trace'], route=route)}
            except _ProviderFailure:
                return {**update, 'failure': ('provider_failed', 'provider')}
            except GenerationError:
                return {**update, 'failure': ('response_rejected', 'validation')}
        return update

    def _retriever(self, state: GraphState) -> dict:
        trace = self._visit(state, 'retriever')
        count = state['attempt_count']
        if not 0 <= count < len(self._plan) <= 2:
            return dict(trace=trace, failure=('internal_error', 'retrieval'))
        strategy = self._plan[count]
        r = state['request']
        scope = state['route_decision'].department_scope if count == 0 and state.get('route_decision') else None
        trace = replace(trace, retrieval_attempts=count+1)
        update = dict(trace=trace, attempt_count=count+1, active_strategy=strategy,
                      retrieved_evidence=(), selected_evidence=(), grading=None, active_scope=scope,
                      prepared=None, generation_result=None,
                      diagnostics=replace(state['diagnostics'], retrieval_strategy=strategy,
                                          evidence=(), warning_codes=()))
        start = perf_counter()
        try:
            self._catalog.check_current(self._catalog.fingerprint)
            options = dict(
                corpus_fingerprint=self._catalog.fingerprint, mode=r.mode, top_k=r.top_k,
                reference_date=r.reference_date)
            if self._profile in (PROFILE, SEMANTIC_PROFILE):
                options['department_scope'] = scope
            hits = tuple(self._stores[strategy].search(r.question, self._embedder, **options))
        except Exception:
            return {**update, 'failure': ('retrieval_failed', 'retrieval')}
        try:
            if len(hits) > r.top_k or len({h.chunk.chunk_id for h in hits}) != len(hits):
                raise ValueError
            for h in hits:
                self._catalog.validate_hit(h, r.mode, r.reference_date)
                if scope is not None and h.chunk.department not in scope:
                    raise ValueError
        except Exception:
            return {**update, 'failure': ('context_failed', 'context')}
        attempt = AttemptTrace(strategy, tuple(h.chunk.chunk_id for h in hits), (), (), perf_counter()-start, scope)
        return {**update, 'retrieved_evidence': hits,
                'trace': replace(trace, attempts=(*trace.attempts, attempt))}

    def _insufficient(self, state: GraphState, trace: GraphTrace, required, reason: str,
                      missing: tuple[str, ...] = (), grading: GradingAssessment | None = None) -> dict:
        assessment = EvidenceAssessment('insufficient', 'reviewed_inputs_missing' if required else 'not_assessed',
                                        (reason,), missing)
        update = dict(assessment=assessment, grading=grading, selected_evidence=(),
                      trace=replace(trace, assessments=(*trace.assessments, assessment)))
        if state['attempt_count'] < len(self._plan):
            return update
        if required:
            return {**update, 'failure': ('context_failed', 'context')}
        return {**update, 'terminal_intent': 'abstain'}

    def _assessor(self, state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        r = state['request']
        trace = self._visit(state, 'assessor')
        required = r.abstention.limitations if r.abstention is not None else r.obligations
        semantic = 'reviewed_inputs_bound' if required else 'not_assessed'
        hits = state['retrieved_evidence']
        grading = None
        try:
            package = self._package(hits, r)
            codes = tuple(w.code for w in package.warnings)
            attempt = replace(trace.attempts[-1], citation_ids=tuple(g.citation.reference_id for g in package.groups),
                              warning_codes=codes)
            trace = replace(trace, attempts=(*trace.attempts[:-1], attempt))
            if any(code in ('budget_excluded', 'source_gap', 'unrelated_context', 'ineligible') for code in codes):
                raise GenerationError('incomplete context')
            available = {s.chunk.chunk_id for group in package.groups for s in group.sources}
            missing = tuple(o.obligation_id for o in required if not set(o.citation.chunk_ids) <= available)
            if missing or not package.groups:
                return self._insufficient(state, trace, required,
                    'missing_reviewed_evidence' if required else 'empty_evidence', missing)
            if runtime.context.workflow is not None:
                request = prepare_grading(r.question, package, model=self._model, limits=self._phase_limits)
                try:
                    response = runtime.context.workflow.request('grader', request, attempt=state['attempt_count'])
                    grading = parse_grading(response.text, package)
                except _ProviderFailure:
                    return dict(trace=trace, failure=('provider_failed', 'provider'))
                except GenerationError:
                    return dict(trace=trace, failure=('response_rejected', 'validation'))
                trace = replace(trace, gradings=(*trace.gradings, grading))
                if grading.unresolved_group_ids:
                    return self._insufficient(state, trace, required, 'unresolved_relevance', grading=grading)
                hits = select_graded_hits(hits, package, grading)
                if not grading.retained_group_ids:
                    return self._insufficient(state, trace, required, 'no_relevant_groups', grading=grading)
                original = package
                package = self._package(hits, r)
                expected = tuple(g.citation for g in original.groups if g.citation.reference_id in grading.retained_group_ids)
                if package.warnings or tuple(g.citation for g in package.groups) != expected:
                    raise GenerationError('grading changed retained source identity')
                available = {s.chunk.chunk_id for group in package.groups for s in group.sources}
                missing = tuple(o.obligation_id for o in required if not set(o.citation.chunk_ids) <= available)
                if missing or not package.groups:
                    return self._insufficient(state, trace, required,
                        'excluded_reviewed_evidence' if missing else 'no_relevant_groups', missing, grading)
            prepared = prepare_answer(r.question, hits, mode=r.mode,
                reference_date=r.reference_date, limits=r.limits, model=self._model,
                corpus_fingerprint=self._catalog.fingerprint, obligations=r.obligations,
                abstention=r.abstention, extractive_scope=r.extractive_scope)
            if prepared.context != package:
                raise GenerationError('context changed during pure preparation')
            assessment = EvidenceAssessment('ready', semantic)
            return dict(prepared=prepared, assessment=assessment, selected_evidence=hits, grading=grading,
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
        phase_b = self._profile == SEMANTIC_PROFILE
        workflow = runtime.context.workflow
        try:
            if state['assessment'].readiness != 'ready' or state['prepared'] is None:
                raise GenerationError('generation requires prepared evidence')
            prepared = state['prepared']
            if phase_b:
                if not isinstance(workflow, PhaseBInvocation):
                    raise GenerationError('Phase B call ledger is unavailable')
                attempt = workflow.generation.calls
                if attempt != state.get('generation_attempt_count', 0) or attempt not in (0, 1):
                    raise GenerationError('Phase B generation count is inconsistent')
                if attempt == 1:
                    assessment = state.get('semantic_assessment')
                    if (assessment is None or assessment.outcome not in ('unsupported', 'incomplete')
                            or not assessment.issue_codes):
                        raise GenerationError('regeneration requires a validated negative assessment')
                    request = prepare_regeneration_request(prepared.request, assessment.issue_codes,
                                                           model=self._model)
                    prepared = replace(prepared, request=request)
            result = generate_answer(prepared, runtime.context.provider)
            if result.prepared is not prepared:
                raise GenerationError('generation changed prepared evidence')
            # Existing render and citation guards are deterministic gates before
            # semantic checking, as well as the final publication boundary.
            projected = _public_answer(result.answer)
            validate_citations(projected.citations, prepared.context)
            if phase_b:
                return {**update, 'prepared': prepared, 'generation_result': result.answer,
                        'generation_attempt_count': workflow.generation.calls,
                        'diagnostics': _prepared_diagnostics(state['diagnostics'], prepared)}
            return {**update, 'generation_result': result.answer}
        except _ProviderFailure:
            if phase_b and isinstance(workflow, PhaseBInvocation):
                update['generation_attempt_count'] = workflow.generation.calls
            return {**update, 'failure': ('provider_failed', 'provider')}
        except SemanticContextError:
            if phase_b and isinstance(workflow, PhaseBInvocation):
                update['generation_attempt_count'] = workflow.generation.calls
            return {**update, 'failure': ('context_failed', 'context')}
        except GenerationError:
            if phase_b and isinstance(workflow, PhaseBInvocation):
                update['generation_attempt_count'] = workflow.generation.calls
            return {**update, 'failure': ('response_rejected', 'validation')}
        except Exception:
            if phase_b and isinstance(workflow, PhaseBInvocation):
                update['generation_attempt_count'] = workflow.generation.calls
            return {**update, 'failure': ('internal_error', 'validation')}

    def _after_generation(self, state: GraphState) -> str:
        if state.get('failure'):
            return 'validator'
        return 'semantic_checker' if self._profile == SEMANTIC_PROFILE else 'validator'

    def _semantic_checker(self, state: GraphState, runtime: Runtime[_Invocation]) -> dict:
        trace = self._visit(state, 'semantic_checker')
        workflow = runtime.context.workflow
        if (self._profile != SEMANTIC_PROFILE or not isinstance(workflow, PhaseBInvocation)
                or state.get('prepared') is None or state.get('generation_result') is None):
            return {'trace': trace, 'failure': ('internal_error', 'validation')}
        answer = state['generation_result']
        try:
            inventory = build_semantic_inventory(answer)
            request = prepare_semantic_check(state['request'].question, answer,
                                             state['prepared'].context, model=self._model)
        except SemanticContextError:
            return {'trace': trace, 'failure': ('context_failed', 'context')}
        except GenerationError:
            return {'trace': trace, 'failure': ('response_rejected', 'validation')}
        attempt = sum(call.purpose == 'semantic_checker' for call in workflow.calls) + 1
        try:
            response = workflow.request('semantic_checker', request, attempt=attempt)
        except _ProviderFailure:
            return {'trace': trace, 'semantic_check_count': sum(
                        call.purpose == 'semantic_checker' for call in workflow.calls),
                    'failure': ('provider_failed', 'provider')}
        except GenerationError:
            return {'trace': trace, 'semantic_check_count': sum(
                        call.purpose == 'semantic_checker' for call in workflow.calls),
                    'failure': ('response_rejected', 'validation')}
        try:
            assessment = parse_semantic_assessment(response.text, inventory)
        except GenerationError:
            workflow.reject_last_response('semantic_checker')
            return {'trace': trace, 'semantic_check_count': sum(
                        call.purpose == 'semantic_checker' for call in workflow.calls),
                    'failure': ('response_rejected', 'validation')}
        if assessment.outcome in ('unsupported', 'incomplete') and state.get('generation_attempt_count') == 1:
            try:
                workflow.arm_regeneration(assessment)
            except GenerationError:
                return {'trace': trace, 'failure': ('internal_error', 'validation')}
        trace = replace(trace, semantic_outcomes=(*trace.semantic_outcomes, assessment.outcome))
        return {'trace': trace, 'semantic_assessment': assessment,
                'semantic_check_count': sum(call.purpose == 'semantic_checker' for call in workflow.calls),
                'regeneration_issue_codes': assessment.issue_codes}

    def _after_semantic(self, state: GraphState) -> str:
        if state.get('failure'):
            return 'validator'
        assessment = state.get('semantic_assessment')
        if assessment is None or assessment.outcome == 'supported':
            return 'validator'
        if (assessment.outcome in ('unsupported', 'incomplete')
                and state.get('generation_attempt_count') == 1
                and state.get('semantic_check_count') == 1):
            return 'generator'
        return 'validator'

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
        workflow = invocation.workflow
        if workflow is not None:
            trace = replace(trace, calls=workflow.calls, usage_unknown=workflow.usage_unknown)
        diagnostics = replace(state['diagnostics'],
                              provider_calls=len(workflow.calls) if workflow else observed.calls,
                              usage=workflow.usage if workflow else observed.usage,
                              elapsed_seconds=perf_counter()-invocation.started)
        failure = state.get('failure')
        if failure is None:
            try:
                r = state['request']
                if not 0 <= state['attempt_count'] <= 2:
                    raise GenerationError('invalid call counters')
                phase_b = self._profile == SEMANTIC_PROFILE
                if workflow is None:
                    if observed.calls > 1:
                        raise GenerationError('bounded generation call limit exceeded')
                elif phase_b:
                    purposes = [call.purpose for call in workflow.calls]
                    if (not isinstance(workflow, PhaseBInvocation) or len(purposes) > 7
                            or purposes.count('router') > 1 or purposes.count('grader') > 2
                            or purposes.count('generator') > 2 or purposes.count('semantic_checker') > 2
                            or any(call.status != 'returned' for call in workflow.calls)):
                        raise GenerationError('Phase B workflow call ledger is invalid')
                elif observed.calls > 1:
                    raise GenerationError('Phase A generation call limit exceeded')
                if state.get('terminal_intent'):
                    if r.obligations or r.abstention or r.extractive_scope or observed.calls:
                        raise GenerationError('terminal template cannot bypass requirements')
                    if phase_b and any(call.purpose in ('generator', 'semantic_checker')
                                       for call in workflow.calls):
                        raise GenerationError('application-authored terminal response cannot follow generation')
                    if workflow is not None and state['terminal_intent'] == 'abstain' and (
                            state['attempt_count'] != len(self._plan) or state['assessment'].readiness != 'insufficient'):
                        raise GenerationError('terminal abstention requires exhausted evidence deficit')
                    answer, package = self._terminal_answer(r, state['terminal_intent'])
                else:
                    prepared = state['prepared']
                    if (prepared is None or prepared.corpus_fingerprint != self._catalog.fingerprint
                            or prepared.context.mode != r.mode or prepared.context.reference_date != r.reference_date
                            or prepared.request.model != self._model):
                        raise GenerationError('publication identity mismatch')
                    if phase_b:
                        assessment = state.get('semantic_assessment')
                        if (assessment is None or assessment.outcome != 'supported'
                                or state.get('generation_attempt_count') not in (1, 2)
                                or state.get('semantic_check_count') != state.get('generation_attempt_count')
                                or observed.calls != state.get('generation_attempt_count')):
                            raise GenerationError('Phase B semantic gate did not pass')
                    elif observed.calls != 1:
                        raise GenerationError('generation call count is inconsistent')
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
                          model: str = MODEL_ID, profile: str = 'bounded_v1',
                          phase_limits: PhaseALimits = PhaseALimits(), budget=None, reservation=None) -> Iterator[GraphQAService]:
    """Explicit opt-in. Reuse one dense adapter/model and local verified catalog.

    snapshot_id identifies an existing immutable local snapshot; never create or
    download one here. Selecting Pinecone retains its existing confirmation gate.
    Injected providers are caller-owned. Setup failures have safe error messages.
    """
    resources = ExitStack()
    try:
        root = validate_project_root(root)
        if budget is not None:
            if reservation is None:
                budget.check_ready()
            else:
                reservation.check_coordinator(budget)
        selected = os.environ.get('VECTOR_STORE', 'chroma') if backend is None else backend
        if (profile not in ('bounded_v1', PROFILE, SEMANTIC_PROFILE)
                or not isinstance(phase_limits, PhaseALimits)
                or (profile in (PROFILE, SEMANTIC_PROFILE)
                    and (provider is None or allow_paid is not False or selected == 'pinecone'))):
            raise QAConfigurationError()
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
                                 strategy_plan=strategy_plan, backend=selected, model=model,
                                 profile=profile, phase_limits=phase_limits, budget=budget)
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
