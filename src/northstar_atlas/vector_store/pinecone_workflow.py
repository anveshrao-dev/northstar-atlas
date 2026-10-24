"""Local frozen snapshots and explicit cloud workflows; no implicit provisioning."""

from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel
from northstar_atlas.local_embeddings import MiniLMProvider, MODEL_ID, MODEL_REVISION
from .pinecone_cloud import CloudError, local_path, preflight, safe_call, validate_description
from .pinecone_state import PublicationState
from .pinecone_store import PineconeStore, PineconeTarget
from .preparation import FrozenSnapshot, corpus_snapshot, freeze_snapshot, prepare_corpus
from .records import IndexConfig, canonical_json, decode_chunk, metadata_for

EXPECTED_COUNTS = {"HR-001": 19, "IT-001": 25, "PLAT-001": 46, "SRE-001": 35, "SEC-001": 34, "COMP-001": 30}
EXPECTED_CORPUS = "20f5ec1d95989e7c6f6b12579d089dea5be0fe815fd5ba357902d195d49dfac9"
SMOKE_QUESTIONS = (
    ("A", "draft_review", "Who reviews and approves a production Terraform plan?"),
    ("B", "draft_review", "Who manages Jenkins running in a physical data center?"),
    ("C", "draft_review", "Can SRE close an incident if Jenkins succeeded but the application is still unhealthy?"),
    ("D", "approved_only", "When must an employee's access be disabled after termination?"),
)


def _snapshot_path(root, namespace):
    try: PublicationState.validate_namespace(namespace)
    except ValueError: raise CloudError("invalid frozen namespace") from None
    return local_path(root, f".local/pinecone/snapshots/{namespace[4:]}.json")


def save_snapshot(root: Path, snapshot: FrozenSnapshot) -> Path:
    path = _snapshot_path(root, snapshot.namespace)
    data = {"snapshot_id": snapshot.snapshot_id, "corpus_fingerprint": snapshot.corpus_fingerprint,
            "config": asdict(snapshot.config), "records": [
                {"id": r.chunk_id, "text": r.chunk.text, "metadata": metadata_for(r.chunk, r.model),
                 "vector": list(r.vector)} for r in snapshot.records]}
    raw = canonical_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != raw: raise CloudError("existing frozen snapshot differs; no overwrite")
    else:
        # Publish a complete, durable file without replacing an existing name.
        # An interrupted temporary write cannot poison the immutable final path.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
    return path


def load_snapshot(root: Path, namespace: str) -> FrozenSnapshot:
    try:
        data = json.loads(_snapshot_path(root, namespace).read_text())
        config_data = data["config"]
        model = EmbeddingModel(**config_data.pop("model"))
        config = IndexConfig(model, **config_data)
        records = [EmbeddedChunk(decode_chunk(r["id"], r["text"], r["metadata"], model),
                                 tuple(r["vector"]), model) for r in data["records"]]
        snapshot = freeze_snapshot(records, config, data["corpus_fingerprint"])
        if snapshot.snapshot_id != data["snapshot_id"] or snapshot.namespace != namespace: raise ValueError
        return snapshot
    except (OSError, ValueError, KeyError, TypeError):
        raise CloudError("frozen snapshot missing or invalid; do not regenerate over an existing generation") from None


def validate_m33(snapshot: FrozenSnapshot, root: Path):
    _, current = corpus_snapshot(root)
    expected_config = IndexConfig(EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, "l2"), 256)
    if (current != EXPECTED_CORPUS or snapshot.corpus_fingerprint != current
            or snapshot.config != expected_config
            or dict(Counter(r.chunk.document_id for r in snapshot.records)) != EXPECTED_COUNTS):
        raise CloudError("corpus counts, hashes or model configuration differ from approved M3.3 snapshot; stop")


def prepare_upload(root: Path) -> tuple[FrozenSnapshot, dict]:
    if corpus_snapshot(root)[1] != EXPECTED_CORPUS: raise CloudError("approved corpus has changed; stop")
    provider = MiniLMProvider(root)
    prepared = prepare_corpus(root, provider)
    if prepared.documents != 6 or prepared.canonical_chunks != 151:
        raise CloudError("unexpected canonical corpus counts; stop")
    snapshot = freeze_snapshot(prepared.records, prepared.config, prepared.fingerprint)
    validate_m33(snapshot, root)
    save_snapshot(root, snapshot)
    return snapshot, {"documents": 6, "canonical_chunks": 151, "embedding_chunks": len(snapshot.records),
                      "chunks_per_document": EXPECTED_COUNTS, "max_tokens": prepared.max_tokens,
                      "warnings": list(prepared.warnings), "embedding_seconds": round(prepared.embedding_seconds, 3),
                      "model_load_seconds": round(provider.load_seconds, 3)}


class RequestMeter:
    """Observe logical HTTP calls and reported usage, never bodies/headers/keys.

    SDK-internal retry attempts and billing totals are not inferred from these counts.
    """
    def __init__(self):
        self.calls = Counter()
        self.read_units = 0
        self.write_units = 0

    def attach(self, client):
        original = client._http
        meter = self

        class ObservedHTTP:
            def __getattr__(self, name):
                function = getattr(original, name)
                if name not in ("get", "post", "patch", "delete", "put"): return function
                def observed(path, **kwargs):
                    operation = path.rsplit("/", 1)[-1]
                    # Never retain arbitrary paths, names or query contents.
                    label = operation if operation in ("upsert", "fetch", "search", "list", "indexes") else "describe"
                    meter.calls[f"{name}:{label}"] += 1
                    response = function(path, **kwargs)
                    try:
                        usage = response.json().get("usage", {})
                        for key in ("read_units", "write_units"):
                            value = usage.get(key, 0)
                            if type(value) in (int, float) and value >= 0:
                                setattr(meter, key, getattr(meter, key) + value)
                    except (ValueError, AttributeError): pass
                    return response
                return observed
        client._http = ObservedHTTP()

    def report(self):
        return {"logical_http_calls": dict(sorted(self.calls.items())), "reported_read_units": self.read_units,
                "reported_write_units": self.write_units,
                "limitation": "only usage fields returned by the API; not total billing or internal retry counts"}


def connect_target(client, meter, settings):
    data = validate_description(safe_call("describe", client.indexes.describe, name=settings.name), settings, require_ready=True)
    target = PineconeTarget(settings.name, data["host"], settings.project_id, settings.cloud, settings.region)
    index = client.index(host=target.host)
    meter.attach(index)
    return target, index


def open_store(root, snapshot, target, client, index):
    validate_m33(snapshot, root)
    return PineconeStore(root, snapshot, target, documents=index.documents,
        describe_target=lambda: client.indexes.describe(name=target.name),
        current_fingerprint=lambda: corpus_snapshot(root)[1])


def active_snapshot(root, target, namespace=None):
    if namespace is not None:
        try: PublicationState.validate_namespace(namespace)
        except ValueError: raise CloudError("invalid explicit namespace; no active-generation fallback") from None
    selected = namespace if namespace is not None else PublicationState(root, asdict(target)).active()
    if selected is None: raise CloudError("no active generation; run explicit index or resume its saved namespace")
    snapshot = load_snapshot(root, selected)
    validate_m33(snapshot, root)
    return snapshot


def status_report(root, settings, client, meter, *, namespace=None):
    """Inspect control/local state even when nothing has ever been published."""
    if namespace is not None:
        PublicationState.validate_namespace(namespace)
    control = preflight(client.indexes, root, settings)
    data = control["description"]
    if data is None: return {"state": "not_created", "remote_audited": False}
    if data["status"]["ready"] is not True:
        return {"state": "initializing", "remote_audited": False}
    target = PineconeTarget(settings.name, data["host"], settings.project_id, settings.cloud, settings.region)
    state = PublicationState(root, asdict(target))
    runs = state.runs()
    overview = {"active_namespace": state.active(), "target": asdict(target), "remote_audited": False,
                "attempts": [{"namespace": key, **{k: value[k] for k in ("state", "phase", "acknowledged", "count") if k in value}}
                             for key, value in sorted(runs.items())]}
    selected = namespace if namespace is not None else state.active()
    if selected is None:
        return {**overview, "state": "incomplete" if runs else "not_indexed"}
    state.validate_namespace(selected)
    run = runs.get(selected)
    if run is None: raise CloudError("unknown namespace; status cannot infer ownership")
    if run.get("state") != "ready" or selected != state.active():
        return {**overview, "state": "incomplete", "namespace": selected}
    snapshot = active_snapshot(root, target, selected)
    with client.index(host=target.host) as index:
        meter.attach(index)
        store = open_store(root, snapshot, target, client, index)
        store.audit()
        return {**overview, **store.status(), "remote_audited": True,
                "corpus_fingerprint": snapshot.corpus_fingerprint, "embedding_config": asdict(snapshot.config)}


def search_report(store, query, provider, *, mode="approved_only", top_k=5, reference_date=None):
    started = perf_counter()
    hits = store.search(query, provider, corpus_fingerprint=store.snapshot.corpus_fingerprint,
                        mode=mode, top_k=top_k, reference_date=reference_date)
    return {"mode": mode, "top_k": top_k, "reference_date": (reference_date or date.today()).isoformat(),
            "latency_seconds": round(perf_counter() - started, 3),
            "score_interpretation": "cosine similarity: higher is closer; distance = 1 - similarity; neither is answer confidence",
            "hits": [{"rank": rank, "id": h.chunk.chunk_id, "citation": h.chunk.citation,
                      "status": h.chunk.status, "source_path": str(h.chunk.source_path),
                      "similarity": h.cosine_similarity, "distance": h.cosine_distance,
                      "required_context": [{"id": c.chunk_id, "citation": c.citation, "status": c.status,
                                            "rank": None, "score": None} for c in h.context]}
                     for rank, h in enumerate(hits, 1)]}


def save_report(root: Path, command: str, report: dict) -> Path:
    if command not in ("preflight", "create", "index", "status", "search", "smoke"):
        raise CloudError("invalid report command")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = local_path(root, f".local/pinecone/m33/{stamp}-{command}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return path
