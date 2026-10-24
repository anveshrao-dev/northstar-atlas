"""Optional SDK-shaped adapter, exercised offline; no client/key/provisioning factory.

The caller injects a Documents API handle and a target-describe function. Calling
methods with a real handle would access Pinecone and needs separate authorization.
Constructors/imports never access credentials, SDK globals, or the network.
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

from northstar_atlas.embeddings import EmbeddingProvider, embed_query
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from .contracts import SearchHit
from .pinecone_state import PublicationState
from .preparation import FrozenSnapshot, STORAGE_SCHEMA_VERSION, float32_vector, freeze_snapshot
from .records import VectorStoreError, canonical_json, context_ids, decode_chunk, digest, metadata_for


@dataclass(frozen=True)
class PineconeTarget:
    name: str
    host: str
    project_id: str
    cloud: str = "azure"
    region: str = "eastus2"

    def __post_init__(self):
        url = urlsplit(self.host)
        if (not self.name or not self.project_id or url.scheme != "https" or not url.hostname
                or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
            raise VectorStoreError("explicit HTTPS target host, index and project identity required")
        if (self.cloud, self.region) != ("azure", "eastus2"):
            raise VectorStoreError("only the selected Azure eastus2 target is authorized by this design")


@dataclass(frozen=True)
class PublicationReport:
    namespace: str
    total: int
    upserted: int
    reused: bool


class _NotVisible(VectorStoreError):
    """Missing records may reflect bounded eventual-consistency delay."""


class _RemoteFailure(VectorStoreError):
    """Transport failure is not by itself proof that stored records are corrupt."""


class PineconeStore:
    def __init__(self, root: Path, snapshot: FrozenSnapshot, target: PineconeTarget, *,
                 documents: Any, describe_target: Callable[[], Any],
                 current_fingerprint: Callable[[], str],
                 delete_namespace: Callable[[str], None] | None = None,
                 poll_attempts: int = 60, poll_interval: float = 5):
        if type(poll_attempts) is not int or not 1 <= poll_attempts <= 60:
            raise VectorStoreError("poll attempts must be between 1 and 60")
        if (isinstance(poll_interval, bool) or not isinstance(poll_interval, Real)
                or not math.isfinite(poll_interval) or not 0 <= poll_interval <= 5):
            raise VectorStoreError("poll interval must be between 0 and 5 seconds")
        self.snapshot = freeze_snapshot(snapshot.records, snapshot.config, snapshot.corpus_fingerprint)
        self.config, self.target = self.snapshot.config, target
        self.namespace = self.snapshot.namespace
        self.documents, self.describe_target = documents, describe_target
        self.current_fingerprint = current_fingerprint
        self.delete_namespace = delete_namespace
        self.poll_attempts, self.poll_interval = poll_attempts, float(poll_interval)
        self.state = PublicationState(root, asdict(target))
        self._audited = False
        self.by_id = {r.chunk_id: r for r in self.snapshot.records}
        self.wire = {r.chunk_id: {"_id": r.chunk_id, "text": r.chunk.text,
                     "embedding": list(r.vector), **metadata_for(r.chunk, r.model),
                     "record_id": r.chunk_id, "snapshot_id": self.snapshot.snapshot_id,
                     "corpus_fingerprint": self.snapshot.corpus_fingerprint,
                     "config_digest": digest(self.config.serialized),
                     "storage_schema_version": STORAGE_SCHEMA_VERSION}
                     for r in self.snapshot.records}
        self.fields = sorted(set(next(iter(self.wire.values()))) - {"_id", "embedding"})

    def _call(self, operation: str, function: Callable, **kwargs):
        try:
            return function(**kwargs)
        except VectorStoreError:
            raise
        except Exception:
            # SDK retries are finite; do not stack an application retry loop or
            # expose exception bodies/headers/query content/credentials.
            raise _RemoteFailure(f"Pinecone {operation} failed; reconcile before retrying") from None

    def _source_check(self, fingerprint: str | None = None):
        expected = self.snapshot.corpus_fingerprint
        if ((fingerprint is not None and fingerprint != expected)
                or self.current_fingerprint() != expected):
            raise VectorStoreError("stale corpus snapshot; prepare the current verified sources")

    def _target_check(self):
        # Pinned SDK 10.0.0 Documents exposes its bound host through _host.
        # Check at every operation, not just construction; no separate client may
        # be described while writes/searches go to a different injected target.
        host = getattr(self.documents, "_host", None)
        if not isinstance(host, str) or host.rstrip("/") != self.target.host.rstrip("/"):
            raise VectorStoreError("Documents transport host differs from selected target")
        description = self._call("describe", self.describe_target)
        if hasattr(description, "to_dict"): description = description.to_dict()
        try:
            field = description["schema"]["fields"]
            vector = field["embedding"]
            deployment = description["deployment"]
            valid = (description["name"] == self.target.name
                     and description["host"].rstrip("/") == self.target.host.rstrip("/")
                     and description["status"]["ready"] is True
                     and description["deletion_protection"] == "enabled"
                     and deployment["deployment_type"] == "managed"
                     and deployment["cloud"] == self.target.cloud
                     and deployment["region"] == self.target.region
                     and set(field) == {"embedding"} and vector["type"] == "dense_vector"
                     and type(vector["dimension"]) is int and vector["dimension"] == self.config.model.dimension
                     and vector["metric"] == "cosine" and not vector.get("embed"))
        except (KeyError, TypeError, AttributeError): valid = False
        if not valid:
            raise VectorStoreError("incompatible or unready Pinecone target/schema; no automatic provisioning")

    def _batches(self) -> list[list[dict]]:
        batches, current = [], []
        for record in self.wire.values():
            key = record["_id"]
            metadata = {k: v for k, v in record.items() if k not in ("_id", "embedding")}
            if not 1 <= len(key) <= 512 or any(not 1 <= ord(c) <= 127 for c in key):
                raise VectorStoreError("record ID exceeds Pinecone ID budget")
            if len(canonical_json(metadata).encode()) > 40_000:
                raise VectorStoreError("record metadata exceeds conservative 40 KB budget; no truncation")
            if current and (len(current) == 50 or
                    len(canonical_json({"documents": current + [record]}).encode()) > 1_000_000):
                batches.append(current)
                current = []
            current.append(record)
            if len(canonical_json({"documents": current}).encode()) > 1_000_000:
                raise VectorStoreError("record exceeds request budget; no truncation")
        if current: batches.append(current)
        return batches

    def _manifest(self, state: str, **extra) -> dict:
        return {"state": state, "target": asdict(self.target), "namespace": self.namespace,
                "snapshot_id": self.snapshot.snapshot_id, "config": self.config.serialized,
                "corpus_fingerprint": self.snapshot.corpus_fingerprint,
                "expected_ids": sorted(self.by_id), "payload_digest": digest(self.wire),
                "source_hashes": {r.chunk.document_id: r.chunk.content_hash for r in self.snapshot.records},
                "count": len(self.by_id), **extra}

    def _owned(self, run: dict) -> None:
        expected = self._manifest("incomplete")
        if any(run.get(k) != v for k, v in expected.items() if k != "state"):
            raise VectorStoreError("namespace ownership/configuration does not match frozen snapshot")

    def _ids(self, namespace: str | None = None, *, expected_count: int | None = None,
             deadline: float | None = None) -> set[str]:
        count = len(self.by_id) if expected_count is None else expected_count
        deadline = time.monotonic() + 300 if deadline is None else deadline
        def read_all():
            found, tokens = set(), set()
            pages = iter(self.documents.list(namespace=namespace or self.namespace, limit=100, timeout=10).pages())
            # Even empty pages consume this bound; a repeated continuation token
            # must never keep a writer locked in an unbounded SDK iterator.
            for _ in range(count + 2):
                if time.monotonic() >= deadline: raise VectorStoreError("pagination deadline exhausted")
                try: page = next(pages)
                except StopIteration: return found
                for record in page.items:
                    if record.id in found or len(found) > count:
                        raise VectorStoreError("invalid duplicate/excessive listing")
                    found.add(record.id)
                token = page.pagination_token
                if token is None: return found
                if token in tokens: raise VectorStoreError("repeated pagination token")
                tokens.add(token)
            raise VectorStoreError("pagination page budget exhausted")
        return self._call("list", read_all)

    def _decode(self, data: dict, *, with_vector: bool):
        data = dict(data)
        key = data.pop("_id", None)
        data.pop("_score", None)
        if key not in self.wire:
            raise VectorStoreError("unexpected record ID in snapshot")
        expected = dict(self.wire[key])
        expected.pop("_id")
        if not with_vector: expected.pop("embedding")
        for field in ("effective_ordinal", "embedding_dimension", "storage_schema_version"):
            value = data.get(field)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value != int(value)):
                raise VectorStoreError("invalid integral metadata field")
            data[field] = int(value)
        if with_vector:
            if "embedding" not in data: raise VectorStoreError("missing vector")
            try: data["embedding"] = list(float32_vector(data["embedding"], self.config))
            except ValueError: raise VectorStoreError("invalid stored vector") from None
        if data != expected:
            raise VectorStoreError("remote payload/text/vector integrity mismatch")
        metadata = {k: data[k] for k in metadata_for(self.by_id[key].chunk, self.config.model)}
        return decode_chunk(key, data["text"], metadata, self.config.model)

    def _fetch(self, ids: set[str], *, siblings: bool = False, deadline: float | None = None) -> dict:
        result = {}
        ordered = sorted(ids)
        for start in range(0, len(ordered), 50):
            if deadline is not None and time.monotonic() >= deadline:
                raise VectorStoreError("verification deadline exhausted")
            batch = ordered[start:start + 50]
            response = self._call("fetch", self.documents.fetch, namespace=self.namespace,
                                  ids=batch, timeout=10)
            if response.namespace != self.namespace or set(response.documents) - set(batch):
                raise VectorStoreError("fetch returned unexpected namespace or IDs")
            if set(response.documents) != set(batch):
                if siblings: raise VectorStoreError("missing required sibling context; cannot return partial evidence")
                raise _NotVisible("missing records during verification")
            for key, document in response.documents.items():
                chunk = self._decode(document.to_dict(), with_vector=True)
                if key != chunk.chunk_id: raise VectorStoreError("fetch ID mismatch")
                result[key] = chunk
        return result

    def _audit(self, deadline: float | None = None):
        ids = self._ids(deadline=deadline)
        if ids - set(self.by_id): raise VectorStoreError("unexpected IDs in immutable namespace")
        if ids != set(self.by_id): raise _NotVisible("missing records during verification")
        self._fetch(ids, deadline=deadline)

    def _dense_search(self, vector, top_k: int, filter: dict):
        response = self._call("search", self.documents.search, namespace=self.namespace, top_k=top_k,
                              score_by=[{"type": "dense_vector", "field": "embedding", "values": list(vector)}],
                              include_fields=self.fields, filter=filter, timeout=10)
        if response.namespace != self.namespace or len(response.matches) > top_k:
            raise VectorStoreError("search returned incompatible namespace or result count")
        return response.matches

    def _wait_ready(self):
        pending = set(self.by_id)
        deadline = time.monotonic() + 300
        for attempt in range(self.poll_attempts):
            if time.monotonic() >= deadline: break
            try:
                self._audit(deadline)
                for key in sorted(pending):
                    if time.monotonic() >= deadline: break
                    found = self._dense_search(self.by_id[key].vector, 1,
                                               {"record_id": {"$eq": key}})
                    if found:
                        chunk = self._decode(found[0].to_dict(), with_vector=False)
                        if chunk.chunk_id != key: raise VectorStoreError("visibility probe returned wrong ID")
                        pending.remove(key)
                if not pending: return
            except _NotVisible:
                pass
            if attempt + 1 < self.poll_attempts: time.sleep(self.poll_interval)
        raise VectorStoreError("query visibility/record verification deadline exhausted; incomplete index")

    def synchronize(self) -> PublicationReport:
        batches = self._batches()  # Preflight every payload before any transport.
        with self.state.lock():
            self._audited = False
            self._source_check()
            self._target_check()
            run = self.state.read_run(self.namespace)
            existing = self._ids()
            if run is None:
                if existing: raise VectorStoreError("remote namespace is not locally owned; no overwrite")
                if sum(r.get("state") != "deleted" for r in self.state.runs().values()) >= 3:
                    raise VectorStoreError("generation limit reached; explicit inactive cleanup required")
            else:
                self._owned(run)
            reused = run is not None and run.get("state") == "ready"
            upserted = 0
            self.state.write_run(self.namespace, self._manifest("incomplete", phase="verification" if reused else "uploading"))
            # Do not repair tampered immutable records implicitly during resume.
            if existing - set(self.by_id): raise VectorStoreError("unexpected IDs in owned namespace")
            self._fetch(existing)
            if not reused:
                for batch in batches:
                    response = self._call("upsert", self.documents.upsert, namespace=self.namespace,
                                          documents=batch, timeout=10)
                    if type(response.upserted_count) is not int or response.upserted_count != len(batch):
                        raise VectorStoreError("partial upsert acknowledgment; incomplete index")
                    upserted += len(batch)
                    self.state.write_run(self.namespace, self._manifest("incomplete", phase="uploading",
                                                                       acknowledged=upserted))
            self._wait_ready()
            self._source_check()
            self.state.write_run(self.namespace, self._manifest("ready", acknowledged=upserted,
                                                               verified_records=len(self.by_id)))
            self.state.publish(self.namespace)
            self._audited = True
            return PublicationReport(self.namespace, len(self.by_id), upserted, reused)

    def status(self) -> dict:
        """Local publication state only: this does not claim a live remote audit."""
        run = self.state.read_run(self.namespace)
        if run is None: return {"state": "incomplete", "namespace": self.namespace, "remote_audited": False}
        self._owned(run)
        return {"state": "ready" if run["state"] == "ready" and self.state.active() == self.namespace else "incomplete",
                "namespace": self.namespace, "count": run["count"], "remote_audited": False}

    def audit(self) -> None:
        """Explicit startup/evaluation integrity audit; reads every remote record."""
        with self.state.lock():
            self._ready()
            self._target_check()
            self._audit_or_invalidate()
            self._source_check()

    def _audit_or_invalidate(self):
        self._audited = False
        try:
            self._audit()
        except VectorStoreError:
            self.state.write_run(self.namespace, self._manifest("incomplete", phase="failed audit"))
            raise
        self._audited = True

    @contextmanager
    def _search_integrity(self):
        try:
            yield
        except _RemoteFailure:
            raise
        except VectorStoreError:
            self._audited = False
            self.state.write_run(self.namespace, self._manifest("incomplete", phase="failed search integrity"))
            raise

    def _ready(self):
        self._source_check()
        if self.status()["state"] != "ready":
            raise VectorStoreError("incomplete or inactive index; synchronize the exact snapshot to recover")

    @staticmethod
    def _eligible(chunk, mode, as_of):
        return (chunk.status in (APPROVED, DRAFT) if mode == "draft_review" else
                chunk.status == APPROVED and chunk.effective_date is not None and chunk.effective_date <= as_of)

    def search(self, query: str, provider: EmbeddingProvider, *, corpus_fingerprint: str,
               mode: RetrievalMode = "approved_only", top_k: int = 5,
               reference_date: date | None = None) -> tuple[SearchHit, ...]:
        if not isinstance(query, str) or not query.strip(): raise VectorStoreError("query must be nonempty")
        if mode not in ("approved_only", "draft_review"): raise VectorStoreError("invalid retrieval mode")
        if type(top_k) is not int or not 1 <= top_k <= 10_000: raise VectorStoreError("invalid top_k")
        if provider.model != self.config.model: raise VectorStoreError("embedding model configuration mismatch")
        as_of = reference_date or date.today()
        if type(as_of) is not date: raise VectorStoreError("invalid reference date")
        with self.state.lock():
            self._source_check(corpus_fingerprint)
            self._ready()
            self._target_check()
            if not self._audited:
                self._audit_or_invalidate()
            vector = float32_vector(embed_query(query, provider).vector, self.config)
            eligibility = ([{"status": {"$eq": APPROVED}}, {"effective_ordinal": {"$gt": 0}},
                            {"effective_ordinal": {"$lte": as_of.toordinal()}}] if mode == "approved_only" else
                           [{"status": {"$in": [APPROVED, DRAFT]}}])
            filter = {"$and": [{"snapshot_id": {"$eq": self.snapshot.snapshot_id}},
                               {"config_digest": {"$eq": digest(self.config.serialized)}}, *eligibility]}
            with self._search_integrity():
                matches = self._dense_search(vector, top_k, filter)
                hits, seen = [], set()
                for document in matches:
                    data = document.to_dict()
                    score = data.get("_score")
                    if (isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score)
                            or not -1.000001 <= score <= 1.000001):
                        raise VectorStoreError("invalid cosine similarity score")
                    chunk = self._decode(data, with_vector=False)
                    if chunk.chunk_id in seen: raise VectorStoreError("duplicate search hit")
                    seen.add(chunk.chunk_id)
                    if not self._eligible(chunk, mode, as_of): raise VectorStoreError("ineligible authority metadata")
                    siblings = self._fetch(set(context_ids(chunk)), siblings=True)
                    if any(not self._eligible(c, mode, as_of) for c in siblings.values()):
                        raise VectorStoreError("ineligible required sibling context")
                    context = tuple(sorted(siblings.values(), key=lambda c: (c.chunk_index, getattr(c, "refinement_index", 0))))
                    hits.append(SearchHit(chunk, 1 - min(1.0, max(-1.0, score)), self.config.model, context))
            self._source_check(corpus_fingerprint)
            return tuple(sorted(hits, key=lambda h: (h.cosine_distance, h.chunk.chunk_id)))

    def cleanup(self, namespace: str, *, approved_namespace: str) -> None:
        """Only an explicitly named, locally owned inactive generation may be removed.

        Deletion requires an explicitly injected transport and exact target approval.
        """
        with self.state.lock():
            self.state.validate_namespace(namespace)
            if namespace != approved_namespace or namespace == self.state.active():
                raise VectorStoreError("cleanup requires exact approval for an inactive namespace")
            run = self.state.read_run(namespace)
            if run is None or run.get("target") != asdict(self.target):
                raise VectorStoreError("cleanup target is not owned")
            if self.delete_namespace is None: raise VectorStoreError("no authorized namespace deletion transport")
            self._target_check()
            count = len(run.get("expected_ids", []))
            if self._ids(namespace, expected_count=count) - set(run.get("expected_ids", [])):
                raise VectorStoreError("unexpected records block cleanup")
            self.state.write_run(namespace, {**run, "state": "deleting"})
            self._call("namespace cleanup", lambda: self.delete_namespace(namespace))
            for attempt in range(self.poll_attempts):
                if not self._ids(namespace, expected_count=count):
                    self.state.write_run(namespace, {**run, "state": "deleted"})
                    return
                if attempt + 1 < self.poll_attempts: time.sleep(self.poll_interval)
            raise VectorStoreError("namespace cleanup incomplete; verify before retrying")
