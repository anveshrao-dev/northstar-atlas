"""Explicit controlled-live opener; no provider or ledger work on import."""
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from .live_budget import GRAPH, PersistentBudget, WorkflowReservation
from .llm import MODEL_ID
from .project import contained_path, validate_project_root
from .qa import QAConfigurationError

# Public corpus identities, not credentials or deployment/account identifiers.
SNAPSHOT_ID = 'e55dcdeca17453b27d45f4d3663ef72b9ba3e635f43689899a8d66b52cf27860'
SNAPSHOT_SHA256 = '6a752f13edef08fd73619f2b63a274ace8058e44592d62c97005fbc6b392ae27'
CORPUS_FINGERPRINT = '20f5ec1d95989e7c6f6b12579d089dea5be0fe815fd5ba357902d195d49dfac9'


def verify_graph_assets(root: Path, *, backend='chroma') -> None:
    """Read existing local state; never download, encode documents or reindex."""
    from .embeddings import EmbeddingModel
    from .local_embeddings import MODEL_ID as EMBEDDING_ID, MODEL_REVISION, MODEL_FILES, model_snapshot
    from .vector_store.catalog import load_catalog
    try:
        if backend not in ('chroma', 'pinecone'):
            raise QAConfigurationError()
        root = validate_project_root(root)
        path = contained_path(root, f'.local/pinecone/snapshots/{SNAPSHOT_ID}.json')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != SNAPSHOT_SHA256:
            raise QAConfigurationError()
        catalog = load_catalog(root, SNAPSHOT_ID)
        expected = EmbeddingModel(model_id=EMBEDDING_ID, revision=MODEL_REVISION,
                                  dimension=384, normalization='l2')
        if (len(catalog.chunks) != 189 or catalog.fingerprint != CORPUS_FINGERPRINT
                or catalog.config.model != expected or catalog.config.token_limit != 256):
            raise QAConfigurationError()
        snapshot = model_snapshot(root)
        for name in MODEL_FILES:
            # Hugging Face's pinned snapshots use links to local cache blobs.
            # Permit those read-only links, but never a target outside this root.
            asset = (snapshot / name).resolve(strict=True)
            if not asset.is_relative_to(root) or not asset.is_file():
                raise QAConfigurationError()
        if backend == 'chroma':
            from .vector_store.store import ChromaStore
            status = ChromaStore(root, catalog.config).status()
            if (status.get('state') != 'ready' or status.get('count') != 189
                    or status.get('actual_count') != 189
                    or status.get('corpus_fingerprint') != catalog.fingerprint):
                raise QAConfigurationError()
        # Pinecone's confirmation, exact active snapshot and remote audit stay
        # in the shared retriever opener, after the workflow is reserved.
    except (OSError, ValueError):
        raise QAConfigurationError() from None


def _check_gate(root, budget, reservation, backend):
    import os
    from .live_preflight import require_private_preview
    from .ui.runtime import graph_operator_authorized
    if (backend not in ('chroma', 'pinecone') or not graph_operator_authorized()
            or (backend == 'pinecone' and os.environ.get('NORTHSTAR_UI_ENABLE_PINECONE') != '1')
            or not isinstance(budget, PersistentBudget)
            or budget.root != root or budget.before_request is not require_private_preview
            or not isinstance(reservation, WorkflowReservation)):
        raise QAConfigurationError()
    reservation.check_setup(budget, profile=GRAPH, backend=backend)
    if reservation.envelope != (7, 184064, 20736):
        raise QAConfigurationError()


class _LiveGraphService:
    """Cache resources only. The caller owns a fresh reservation on every ask."""
    def __init__(self, root, budget, resources, backend):
        self.root, self.budget, self.resources = root, budget, resources
        self.backend = backend
        self.service = None

    def ask(self, question, *, mode='approved_only', reference_date=None, reservation=None, **kwargs):
        from .live_preflight import require_private_preview
        _check_gate(self.root, self.budget, reservation, self.backend)
        as_of = reference_date or datetime.now(timezone.utc).date()
        reservation.check_request(profile=GRAPH, question=question, mode=mode,
                                  backend=self.backend, reference_date=as_of)
        # Repeat local and access checks for cached resources too, before routing.
        if self.backend == 'chroma':
            verify_graph_assets(self.root)
        else:
            verify_graph_assets(self.root, backend=self.backend)
        require_private_preview()
        _check_gate(self.root, self.budget, reservation, self.backend)  # Access checks may outlive pricing.
        if self.service is None:
            from .nebius import NebiusProvider, load_settings
            from .orchestration import open_graph_qa_service
            settings = load_settings(self.root)
            if settings.model != MODEL_ID:
                raise QAConfigurationError()
            with ExitStack() as opening:
                provider = opening.enter_context(NebiusProvider(settings, allow_paid=True,
                    max_attempts=1, requires_budget=True))
                service = opening.enter_context(open_graph_qa_service(self.root,
                    snapshot_id=SNAPSHOT_ID, backend=self.backend, profile=GRAPH,
                    provider=provider, allow_paid=False, model=MODEL_ID,
                    budget=self.budget, reservation=reservation))
                self.resources.enter_context(opening.pop_all())
                self.service = service
        return self.service.ask(question, mode=mode, reference_date=as_of,
                                reservation=reservation, **kwargs)


@contextmanager
def open_live_graph_service(root: Path, *, backend='chroma', profile=GRAPH,
                            allow_paid=False, model=MODEL_ID, budget=None, reservation=None):
    """Only an explicitly authorized, fully reserved semantic graph invocation.

    allow_paid remains False at the injection-only graph factory boundary. Live
    authorization comes from server gates plus the durable invocation permit,
    never from a caller flipping the generic factory's paid flag.
    """
    root = validate_project_root(root)
    if backend not in ('chroma', 'pinecone') or profile != GRAPH or model != MODEL_ID or allow_paid is not False:
        raise QAConfigurationError()
    _check_gate(root, budget, reservation, backend)
    with ExitStack() as resources:
        yield _LiveGraphService(root, budget, resources, backend)
