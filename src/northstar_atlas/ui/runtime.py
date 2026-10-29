"""Public projections, submission accounting and ownership of a reusable QAService."""
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
import os
from threading import Lock
from typing import Callable

from northstar_atlas.evidence import CitationReference
from northstar_atlas.ingestion.document_loader import DRAFT, RetrievalMode
from northstar_atlas.llm import GenerationLimits, MODEL_ID
from northstar_atlas.qa import PublicPresentation, QAFailure, QASuccess, open_qa_service, open_retriever
from northstar_atlas.project import validate_project_root
from northstar_atlas.vector_store.contracts import SearchHit, score_fields
from northstar_atlas.vector_store.catalog import eligible
from northstar_atlas.vector_store.records import IndexConfig, context_ids, metadata_for, validate_context

LIMITS = GenerationLimits(context_tokens=32768, output_tokens=2048, margin_tokens=512)
MAX_ATTEMPTS = 5
MAX_QUESTION_CHARS = 2000
STANDARD_WORKFLOW = 'standard'
LANGGRAPH_WORKFLOW = 'langgraph_semantic'
LANGGRAPH_PROFILE = 'llm_routing_grading_semantic_v1'
POLICY_SEARCH = 'Policy search — retrieval only'


def graph_operator_authorized() -> bool:
    """Static operator gate only; reservation/assets/access are checked on submit."""
    return all(os.environ.get(name) == '1' for name in (
        'NORTHSTAR_CONTROLLED_LIVE', 'NORTHSTAR_UI_ENABLE_LIVE',
        'NORTHSTAR_UI_ENABLE_GRAPH', 'NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED'))


def _live_graph_opener(*args, **kwargs):
    # Keep optional graph/model modules out of the UI-only offline environment.
    from northstar_atlas.live_graph import open_live_graph_service
    return open_live_graph_service(*args, **kwargs)

# Only fixed public errors cross this boundary, never exception strings or diagnostics.
ERRORS = {
    'invalid_input': 'Please enter a question of 1–2,000 characters and valid retrieval options.',
    'configuration_failed': 'The question-answering service is not configured for this request.',
    'retrieval_failed': 'Policy evidence could not be retrieved. No answer was generated.',
    'context_failed': 'Complete, eligible evidence could not be prepared within the configured limits.',
    'provider_failed': 'The answer provider could not complete the request. No answer was accepted.',
    'response_rejected': 'The generated response did not pass validation and has been withheld.',
    'internal_error': 'The request could not be completed. No answer was accepted.',
    'opt_in_required': 'Live answers require server authorization and your paid-inference opt-in.',
    'workflow_disabled': 'LangGraph live execution is disabled until shared durable budget protection is available. No request was sent.',
    'workflow_backend_unsupported': 'LangGraph is not available with this retrieval backend. No request was sent.',
    'backend_disabled': 'This retrieval backend is not enabled. No request was sent.',
    'duplicate': 'This submission has already been attempted. It will not be sent again.',
    'busy': 'Another request is running. No request was queued; submit again when it finishes.',
    'budget_unavailable': 'The shared live budget is unavailable or this request is not authorized. No answer was accepted.',
    'batch_exhausted': 'The approved five-question batch is exhausted. A separately reviewed batch approval is required.',
    'budget_exhausted': 'The five-attempt allowance for this server process has been reached.',
    'interrupted': 'This submission was interrupted or its outcome is unknown. It will not be retried automatically.',
    'cleanup_failed': 'Resources could not be fully disconnected. Restart the server before live use.',
    'search_failed': 'Verified policy evidence could not be retrieved. Check the existing local index and cached model, or try a shorter question. No LLM inference was attempted.',
}


@dataclass(frozen=True)
class PublicView:
    kind: str
    text: str
    citations: tuple[str, ...] = ()
    coverage: str = 'not_assessed'
    proposed: bool = False
    code: str | None = None
    presentation: PublicPresentation | None = None
    sources: tuple[CitationReference, ...] = ()
    passages: tuple[SearchHit, ...] = ()


def error_view(code: str) -> PublicView:
    code = code if code in ERRORS else 'internal_error'
    return PublicView('error', ERRORS[code], code=code)


def project_result(result: QASuccess | QAFailure, mode: RetrievalMode) -> PublicView:
    """Consume only the existing public contract; no UI revalidation of raw output."""
    if isinstance(result, QAFailure):
        return error_view(result.public.code)
    if not isinstance(result, QASuccess) or result.public.kind not in ('answer', 'clarify', 'abstain'):
        return error_view('response_rejected')
    public = result.public
    proposed = any(c.status == DRAFT for c in public.citations)
    if mode not in ('approved_only', 'draft_review') or (mode == 'approved_only' and proposed):
        return error_view('response_rejected')
    try:
        sources = tuple(dict.fromkeys(public.citations))
        labels = tuple(c.label for c in sources)
        presentation = public.presentation
        if presentation is not None:
            # Public-shape consistency only: never reinterpret or revalidate policy claims.
            referenced = tuple(c for s in presentation.statements for c in s.citations) + tuple(
                q.citation for q in presentation.quotations)
            if any(c not in sources for c in referenced):
                return error_view('response_rejected')
    except ValueError:
        return error_view('response_rejected')
    return PublicView(public.kind, public.text, labels, public.coverage.status, proposed,
                      presentation=presentation, sources=sources)


def project_search(hits: tuple[SearchHit, ...], mode: RetrievalMode, as_of: date) -> PublicView:
    """Expose exact verified passages, never an answer or a relevance threshold.

    Source authenticity/publication checks belong to open_retriever/store.search.
    Repeat eligibility and complete-family checks before publishing any passage.
    """
    try:
        if (mode not in ('approved_only', 'draft_review') or type(as_of) is not date
                or not isinstance(hits, tuple) or len(hits) > 5):
            raise ValueError('invalid search result')
        direct_ids, sources = set(), {}
        for hit in hits:
            if not isinstance(hit, SearchHit) or hit.chunk.chunk_id in direct_ids:
                raise ValueError('invalid dense result')
            direct_ids.add(hit.chunk.chunk_id)
            score_fields(hit)
            family = (hit.chunk, *hit.context)
            if set(context_ids(hit.chunk)) != {c.chunk_id for c in hit.context}:
                raise ValueError('unexpected sibling context')
            validate_context(family, IndexConfig(hit.model, getattr(hit.chunk, 'token_limit', 256)))
            for chunk in family:
                metadata_for(chunk, hit.model)
                if (not eligible(chunk, mode, as_of)
                        or (chunk.chunk_id in sources and sources[chunk.chunk_id] != chunk)):
                    raise ValueError('ineligible or conflicting evidence')
                sources[chunk.chunk_id] = chunk
        labels = tuple(dict.fromkeys(c.citation for c in sources.values()))
        return PublicView('evidence', 'Retrieved policy passages — no LLM answer generated.' if hits
                          else 'No eligible passages were returned. This does not establish that no applicable policy exists.',
                          citations=labels, proposed=any(c.status == DRAFT for c in sources.values()),
                          passages=hits)
    except Exception:
        return error_view('search_failed')


@dataclass(frozen=True)
class Submission:
    turn_id: str
    question: str
    mode: RetrievalMode
    backend: str
    reference_date: date
    demo: bool = False
    workflow: str = STANDARD_WORKFLOW
    retrieval_only: bool = False

    def valid(self) -> bool:
        return (bool(self.turn_id) and isinstance(self.question, str)
                and 0 < len(self.question.strip()) <= MAX_QUESTION_CHARS
                and self.mode in ('approved_only', 'draft_review')
                and self.backend in ('chroma', 'pinecone')
                and self.workflow in (STANDARD_WORKFLOW, LANGGRAPH_WORKFLOW)
                and type(self.retrieval_only) is bool
                and (not self.retrieval_only or (not self.demo and self.workflow == STANDARD_WORKFLOW))
                and type(self.reference_date) is date)


class ResourceOwner:
    """One serialized backend context per process. Counters survive disconnect.

    Construction does not load credentials, MiniLM or databases. The opener runs
    only for a gated submission. It preserves QAService's single provider attempt.
    """
    def __init__(self, root: Path, *, live_enabled: bool = False,
                 pinecone_enabled: bool = False, opener: Callable | None = None,
                 graph_enabled: bool = False, graph_opener: Callable | None = None, budget=None,
                 search_opener: Callable | None = None):
        self.root = validate_project_root(root)
        self.live_enabled, self.pinecone_enabled = live_enabled, pinecone_enabled
        self._opener = open_qa_service if opener is None else opener
        self._search_opener = open_retriever if search_opener is None else search_opener
        self.graph_enabled = graph_enabled
        self._controlled_graph = graph_opener is None
        self._graph_opener = _live_graph_opener if graph_opener is None else graph_opener
        self._budget = budget
        self.lock = Lock()
        self._stack = ExitStack()
        self._service = None
        self._backend = None
        self._workflow = None
        self._broken = False
        self._seen: set[str] = set()
        self.attempts = 0

    def _disconnect(self) -> None:
        self._service, self._backend, self._workflow = None, None, None
        try:
            self._stack.close()
        except Exception:
            self._broken = True
        self._stack = ExitStack()

    def close(self) -> bool:
        """Nonblocking, idempotent cleanup; never interrupt an in-flight ask."""
        if not self.lock.acquire(blocking=False):
            return False
        try:
            self._disconnect()
            return not self._broken
        finally:
            self.lock.release()

    def batch_remaining(self) -> int | None:
        """Safe display-only status. None means unavailable, never a fresh allowance."""
        try:
            from northstar_atlas.demo_batch import activation, remaining
            from northstar_atlas.live_budget import configured_budget
            plan = activation()
            if self._budget is None:
                self._budget = configured_budget(self.root)
            return remaining(self._budget, plan)
        except Exception:
            return None

    def submit(self, request: Submission, *, consent: bool) -> PublicView:
        if not request.valid() or request.demo:
            return error_view('invalid_input')
        if request.retrieval_only:
            return self._search_submit(request)
        if not self.live_enabled or consent is not True:
            return error_view('opt_in_required')
        if (request.workflow == LANGGRAPH_WORKFLOW
                and (not self.graph_enabled
                     or (self._controlled_graph and not graph_operator_authorized()))):
            return error_view('workflow_disabled')
        if request.backend == 'pinecone' and not self.pinecone_enabled:
            return error_view('backend_disabled')
        if not self.lock.acquire(blocking=False):
            return error_view('busy')
        try:
            if self._broken:
                return error_view('cleanup_failed')
            if request.turn_id in self._seen:
                return error_view('duplicate')
            batch_enabled = os.environ.get('NORTHSTAR_DEMO_BATCH_ENABLED', '0') != '0'
            if batch_enabled and request.workflow != STANDARD_WORKFLOW:
                return error_view('workflow_disabled')
            if batch_enabled and request.backend != 'pinecone':
                return error_view('backend_disabled')
            if batch_enabled:
                remaining = self.batch_remaining()
                if remaining is None:
                    return error_view('budget_unavailable')
                if remaining == 0:
                    return error_view('batch_exhausted')
            if not batch_enabled and self.attempts >= MAX_ATTEMPTS:
                return error_view('budget_exhausted')
            self._seen.add(request.turn_id)
            self.attempts += 1  # Conservative: setup/provider/validation failures also consume an attempt.
            from northstar_atlas.live_budget import BudgetError, WorkflowRequest, LINEAR, GRAPH, configured_budget
            try:
                if self._budget is None and os.environ.get('NORTHSTAR_CONTROLLED_LIVE') == '1':
                    self._budget = configured_budget(self.root)
                if batch_enabled:
                    from northstar_atlas.demo_batch import run_question
                    self._disconnect()
                    if self._broken:
                        return error_view('cleanup_failed')
                    result = run_question(self._budget, WorkflowRequest(request.turn_id, LINEAR,
                        request.question, request.mode, request.backend, request.reference_date),
                        opener=self._opener, limits=LIMITS)
                    return project_result(result, request.mode)
                scope = nullcontext(None) if self._budget is None else self._budget.reserve(WorkflowRequest(
                    request.turn_id, GRAPH if request.workflow == LANGGRAPH_WORKFLOW else LINEAR,
                    request.question, request.mode, request.backend, request.reference_date))
                with scope as reservation:
                    result = self._execute(request, reservation)
                return result
            except (BudgetError, OSError, ValueError):
                return error_view('budget_unavailable')
            except Exception:
                return error_view('internal_error')
        finally:
            self.lock.release()

    def _search_submit(self, request: Submission) -> PublicView:
        """Local retrieval only: no QAService, provider, live counter or budget IO."""
        if request.backend != 'chroma':
            return error_view('backend_disabled')
        if not self.lock.acquire(blocking=False):
            return error_view('busy')
        try:
            if self._broken:
                return error_view('cleanup_failed')
            if request.turn_id in self._seen:
                return error_view('duplicate')
            self._seen.add(request.turn_id)
            try:
                # Recheck current source hashes even when model/store resources are cached.
                from northstar_atlas.vector_store.preparation import corpus_snapshot
                _, current = corpus_snapshot(self.root)
                if self._service is None or self._workflow != POLICY_SEARCH:
                    self._disconnect()
                    if self._broken:
                        return error_view('cleanup_failed')
                    self._service = self._stack.enter_context(self._search_opener(self.root, 'chroma'))
                    self._backend, self._workflow = 'chroma', POLICY_SEARCH
                store, embedder, fingerprint = self._service
                if fingerprint != current:
                    raise ValueError('source corpus changed')
                hits = store.search(request.question, embedder, corpus_fingerprint=current,
                                    mode=request.mode, top_k=5, reference_date=request.reference_date)
                if any(hit.model != embedder.model for hit in hits):
                    raise ValueError('embedding identity mismatch')
                return project_search(hits, request.mode, request.reference_date)
            except Exception:
                self._disconnect()
                return error_view('search_failed')
        finally:
            self.lock.release()


    def _execute(self, request: Submission, reservation) -> PublicView:
        if (self._service is None or request.backend != self._backend
                or request.workflow != self._workflow):
            self._disconnect()
            if self._broken:
                return error_view('cleanup_failed')
            try:
                if request.workflow == LANGGRAPH_WORKFLOW:
                    opener = self._graph_opener
                    options = dict(backend=request.backend, profile=LANGGRAPH_PROFILE,
                                  allow_paid=False, model=MODEL_ID)
                else:
                    opener = self._opener
                    options = dict(backend=request.backend, allow_paid=True, model=MODEL_ID)
                if self._budget is not None:
                    options.update(budget=self._budget, reservation=reservation)
                self._service = self._stack.enter_context(opener(self.root, **options))
                self._backend = request.backend
                self._workflow = request.workflow
            except Exception:
                self._disconnect()
                return error_view('configuration_failed')
        try:
            options = {} if reservation is None else {'reservation': reservation}
            result = self._service.ask(request.question, mode=request.mode, top_k=5,
                                       reference_date=request.reference_date, limits=LIMITS, **options)
            return project_result(result, request.mode)
        except Exception:
            return error_view('internal_error')

@dataclass(frozen=True)
class ChatTurn:
    submission: Submission
    view: PublicView


@dataclass
class Conversation:
    """Session-local display history, never a provider's conversation context."""
    history: list[ChatTurn] = field(default_factory=list)
    pending: Submission | None = None
    started: set[str] = field(default_factory=set)

    def queue(self, request: Submission) -> bool:
        if self.pending is not None or request.turn_id in self.started:
            return False
        self.pending = request
        return True

    def run_pending(self, owner: ResourceOwner, *, consent: bool) -> None:
        request = self.pending
        if request is None:
            return
        if request.turn_id in self.started:
            view = error_view('interrupted')
        else:
            self.started.add(request.turn_id)  # Write before any possible side effect.
            if request.demo:
                from northstar_atlas.ui.demo import DemoQAService
                view = project_result(DemoQAService().ask(request.question, mode=request.mode), request.mode)
            else:
                view = owner.submit(request, consent=consent)
        self.history.append(ChatTurn(request, view))
        self.pending = None

    def clear(self) -> None:
        if self.pending is None:
            self.history.clear()  # Keep submission IDs and process accounting.
