"""Small persistent Chroma adapter with fail-closed publication and local locking."""

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import tomllib

from northstar_atlas.embeddings import EmbeddedChunk, EmbeddingModel, EmbeddingProvider, embed_query, validate_vector
from northstar_atlas.ingestion.chunker import PolicyChunk
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from .records import IndexConfig, VectorStoreError, canonical_json, context_ids, decode_chunk, digest, metadata_for, validate_context
from .contracts import SearchHit  # Compatibility export for existing callers.


COLLECTION_NAME = "northstar_atlas_policies_v1"


def _vector_digest(vectors: dict[str, Sequence[float]]) -> str:
    """Hash exactly the float32 representation Chroma persists, in ID order."""
    packed = {}
    for key, vector in vectors.items():
        try:
            raw = struct.pack(f"<{len(vector)}f", *vector)
            rounded = struct.unpack(f"<{len(vector)}f", raw)
        except (OverflowError, struct.error):
            raise VectorStoreError("vector cannot be represented as Chroma float32 values") from None
        if not all(math.isfinite(x) for x in rounded) or math.hypot(*rounded) == 0:
            raise VectorStoreError("vector is nonfinite or zero in Chroma float32 storage")
        packed[key] = raw.hex()
    return digest(packed)


@dataclass(frozen=True)
class IndexReport:
    inserted: int
    updated: int
    unchanged: int
    deleted: int
    total: int
    changed_documents: tuple[str, ...]


def _eligible(chunk: PolicyChunk, mode: RetrievalMode, as_of: date) -> bool:
    return (chunk.status in (APPROVED, DRAFT) if mode == "draft_review" else
            chunk.status == APPROVED and chunk.effective_date is not None and chunk.effective_date <= as_of)


class ChromaStore:
    """Low-level adapter. Corpus workflows supply a freshly verified fingerprint.

    An advisory lock serializes this adapter's operations across local processes.
    Direct Chroma clients and manual filesystem edits must not run concurrently.
    """

    def __init__(self, project_root: Path, config: IndexConfig):
        self.root = Path(project_root).resolve(strict=True)
        project = self.root / "pyproject.toml"
        if not project.resolve().is_relative_to(self.root):
            raise VectorStoreError("project configuration resolves outside root")
        if tomllib.loads(project.read_text())["project"]["name"] != "northstar-atlas":
            raise VectorStoreError("expected a Northstar Atlas project root")
        self.config = config
        self.path = self.root / ".local/chroma"
        self.state_path = self.path / "index-state.json"

    def _check_paths(self) -> None:
        for path in (self.root / ".local", self.path):
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise VectorStoreError("database/cache path is a symlink or resolves outside project")
        for path in self.path.rglob("*"):
            if path.is_symlink():
                raise VectorStoreError("database contains a symlink; no writes permitted")

    @contextmanager
    def _lock(self):
        self._check_paths()
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / ".index.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise VectorStoreError("index is busy with another local operation; retry when it finishes") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _collection(self, *, create: bool = False):
        import chromadb
        from chromadb.config import Settings
        from chromadb.errors import NotFoundError

        client = chromadb.PersistentClient(path=str(self.path), settings=Settings(
            _env_file=None, chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
            anonymized_telemetry=False, chroma_otel_granularity="none",
            chroma_otel_collection_endpoint="", allow_reset=False,
        ))
        try:
            collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        except NotFoundError:
            if not create:
                raise VectorStoreError("incomplete index: collection missing; rerun index") from None
            collection = client.create_collection(COLLECTION_NAME, embedding_function=None,
                metadata={"index_config": self.config.serialized},
                configuration={"hnsw": {"space": "cosine"}})
        # Inspect raw config without constructing a stored embedding function.
        # Chroma 1.5.9 serializes explicit embedding_function=None as type=legacy.
        configuration = collection.configuration_json
        if ((collection.metadata or {}).get("index_config") != self.config.serialized
                or configuration.get("hnsw", {}).get("space") != "cosine"
                or configuration.get("embedding_function") not in (None, {"type": "legacy"})):
            raise VectorStoreError("incompatible collection model/configuration; do not mix embedding spaces")
        return collection

    def _read_state(self) -> dict:
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError
            return state
        except (FileNotFoundError, ValueError):
            return {"state": "incomplete", "reason": "missing or invalid index state; rerun index"}

    def _write_state(self, state: dict) -> None:
        # Atomic sidecar replacement; ready is published only after durable Chroma calls.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path,
                                         prefix="index-state-", suffix=".tmp", delete=False) as stream:
            stream.write(canonical_json(state) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, self.state_path)
        directory = os.open(self.path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _records(collection) -> dict[str, tuple[str, dict]]:
        result = collection.get(include=["documents", "metadatas"])
        return {key: (text, metadata) for key, text, metadata in
                zip(result["ids"], result["documents"], result["metadatas"], strict=True)}

    def _stored_vectors(self, collection) -> dict[str, tuple[float, ...]]:
        result = collection.get(include=["embeddings"])
        try:
            vectors = {key: validate_vector(vector.tolist(), dimension=self.config.model.dimension)
                       for key, vector in zip(result["ids"], result["embeddings"], strict=True)}
            return vectors
        except (ValueError, TypeError, AttributeError):
            raise VectorStoreError("incomplete index: invalid stored vectors; rerun index") from None

    def _ready(self, collection, state: dict) -> dict[str, PolicyChunk]:
        if state.get("state") != "ready":
            raise VectorStoreError("incomplete index; rerun index to recover before searching")
        if state.get("config") != self.config.serialized:
            raise VectorStoreError("incompatible index model/configuration")
        records = self._records(collection)
        if len(records) != state.get("count") or digest(records) != state.get("record_digest"):
            raise VectorStoreError("incomplete index: record integrity check failed; rerun index")
        if _vector_digest(self._stored_vectors(collection)) != state.get("vector_digest"):
            raise VectorStoreError("incomplete index: vector integrity check failed; rerun index")
        chunks = {key: decode_chunk(key, text, meta, self.config.model) for key, (text, meta) in records.items()}
        validate_context(tuple(chunks.values()), self.config)
        return chunks

    def synchronize(self, records: Sequence[EmbeddedChunk], *, corpus_fingerprint: str) -> IndexReport:
        """Preflight first; mark incomplete, upsert, prune, verify, then publish ready."""
        if not records or not isinstance(corpus_fingerprint, str) or not corpus_fingerprint:
            raise VectorStoreError("a nonempty verified corpus and fingerprint are required")
        vectors = []
        wanted = {}
        for record in records:
            if record.model != self.config.model:
                raise VectorStoreError("embedding model/configuration mismatch")
            vector = validate_vector(record.vector, dimension=self.config.model.dimension)
            if self.config.model.normalization == "l2" and not math.isclose(math.hypot(*vector), 1.0, rel_tol=1e-5):
                raise VectorStoreError("vector does not match declared l2 normalization")
            vectors.append(vector)
            wanted[record.chunk_id] = (record.chunk.text, metadata_for(record.chunk, record.model))
        validate_context(tuple(r.chunk for r in records), self.config)
        expected_vectors = {r.chunk_id: v for r, v in zip(records, vectors, strict=True)}
        _vector_digest(expected_vectors)  # Reject values not representable in float32 before writing.
        with self._lock():
            collection = self._collection(create=True)
            previous = self._records(collection)
            # Corrupt/foreign old metadata must not prevent repair of this owned collection.
            old_hashes = {meta["document_id"]: meta["content_hash"] for _, meta in previous.values()
                          if isinstance(meta, dict) and isinstance(meta.get("document_id"), str)
                          and isinstance(meta.get("content_hash"), str)}
            new_hashes = {meta["document_id"]: meta["content_hash"] for _, meta in wanted.values()}
            common = set(previous) & set(wanted)
            unchanged = sum(previous[key] == wanted[key] for key in common)
            report = IndexReport(len(set(wanted) - set(previous)), len(common) - unchanged, unchanged,
                len(set(previous) - set(wanted)), len(wanted),
                tuple(sorted(k for k in set(old_hashes) | set(new_hashes) if old_hashes.get(k) != new_hashes.get(k))))
            state = {"state": "incomplete", "config": self.config.serialized,
                     "corpus_fingerprint": corpus_fingerprint, "count": len(wanted)}
            self._write_state(state)
            try:
                # Bounded batches stay below Chroma's request limit; order is source order.
                for start in range(0, len(records), 100):
                    batch = records[start:start + 100]
                    collection.upsert(ids=[r.chunk_id for r in batch],
                        documents=[r.chunk.text for r in batch],
                        metadatas=[wanted[r.chunk_id][1] for r in batch],
                        embeddings=[list(v) for v in vectors[start:start + 100]])
                obsolete = sorted(set(previous) - set(wanted))
                for start in range(0, len(obsolete), 100):
                    collection.delete(ids=obsolete[start:start + 100])
                if self._records(collection) != wanted:
                    raise VectorStoreError("persisted records do not match the complete input")
                stored_vectors = self._stored_vectors(collection)
                if set(stored_vectors) != set(expected_vectors):
                    raise VectorStoreError("persisted vector IDs do not match the validated input")
                for key, expected in expected_vectors.items():
                    actual = stored_vectors[key]
                    expected_norm, actual_norm = math.hypot(*expected), math.hypot(*actual)
                    # Cosine storage may normalize/re-round float32 coordinates.
                    # Verify direction numerically, then fingerprint the actual persisted bytes.
                    if any(not math.isclose(a / actual_norm, e / expected_norm, rel_tol=1e-5, abs_tol=1e-6)
                           for a, e in zip(actual, expected, strict=True)):
                        raise VectorStoreError("persisted vector differs from validated input direction")
                state.update(state="ready", record_digest=digest(wanted),
                             vector_digest=_vector_digest(stored_vectors))
                self._write_state(state)
            except VectorStoreError as exc:
                raise VectorStoreError(f"index incomplete: {exc}; rerun index to recover") from None
            except Exception as exc:
                # The incomplete marker already exists, including after process termination.
                raise VectorStoreError(f"index incomplete ({type(exc).__name__}); rerun index to recover") from None
            return report

    def status(self) -> dict:
        self._check_paths()
        if not self.path.exists():
            return {"state": "absent", "count": 0}
        with self._lock():
            state = self._read_state()
            try:
                collection = self._collection()
                state["actual_count"] = collection.count()
                self._ready(collection, state)
            except VectorStoreError as exc:
                state = {**state, "state": "incomplete", "reason": str(exc)}
            return state

    def search(self, query: str, provider: EmbeddingProvider, *, corpus_fingerprint: str,
               mode: RetrievalMode = "approved_only", top_k: int = 5,
               reference_date: date | None = None) -> tuple[SearchHit, ...]:
        if not isinstance(query, str) or not query.strip():
            raise VectorStoreError("query must be a nonempty string")
        if mode not in ("approved_only", "draft_review") or type(top_k) is not int or not 1 <= top_k <= 100:
            raise VectorStoreError("invalid retrieval mode or top_k (must be 1..100)")
        if provider.model != self.config.model:
            raise VectorStoreError("query embedding model/configuration mismatch")
        if hasattr(provider, "max_input_tokens") and provider.max_input_tokens != self.config.token_limit:
            raise VectorStoreError("query token limit configuration mismatch")
        as_of = reference_date or datetime.now(timezone.utc).date()
        self._check_paths()
        if not self.path.exists():
            raise VectorStoreError("index absent; run index first")
        with self._lock():
            state = self._read_state()
            collection = self._collection()
            chunks = self._ready(collection, state)
            if state.get("corpus_fingerprint") != corpus_fingerprint:
                raise VectorStoreError("stale index: corpus changed; rerun index")
            eligible = {key: c for key, c in chunks.items() if _eligible(c, mode, as_of)}
            if not eligible:
                return ()
            query_vector = embed_query(query, provider)
            if query_vector.model != self.config.model:
                raise VectorStoreError("query embedding model changed; configuration mismatch")
            where = ({"$and": [{"status": APPROVED}, {"effective_ordinal": {"$gt": 0}},
                       {"effective_ordinal": {"$lte": as_of.toordinal()}}]} if mode == "approved_only"
                     else {"status": {"$in": [APPROVED, DRAFT]}})
            result = collection.query(query_embeddings=[list(query_vector.vector)],
                                      n_results=min(top_k, len(eligible)), where=where, include=["distances"])
            hits = []
            for key, distance in zip(result["ids"][0], result["distances"][0], strict=True):
                if key not in eligible or not math.isfinite(distance):
                    raise VectorStoreError("invalid or ineligible search result")
                chunk = eligible[key]
                siblings = []
                for sibling_id in context_ids(chunk):
                    if sibling_id not in eligible:
                        raise VectorStoreError("required sibling context missing or ineligible; no partial evidence returned")
                    siblings.append(eligible[sibling_id])
                siblings.sort(key=lambda c: (c.chunk_index, getattr(c, "refinement_index", 0)))
                hits.append(SearchHit(chunk, float(distance), self.config.model, tuple(siblings)))
            return tuple(sorted(hits, key=lambda h: (h.cosine_distance, h.chunk.chunk_id)))
