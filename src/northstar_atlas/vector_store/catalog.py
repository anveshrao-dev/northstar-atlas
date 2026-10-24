"""Verified local chunk catalog shared by lexical ranking and fusion."""
from collections import defaultdict
from dataclasses import fields
from datetime import date, datetime, timezone
from pathlib import Path
import re
from types import MappingProxyType

from northstar_atlas.ingestion.chunker import PolicyChunk, chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT, RetrievalMode
from northstar_atlas.ingestion.refiner import RefinedPolicyChunk, reconstruct
from northstar_atlas.project import validate_project_root
from .contracts import RetrievalHit, score_fields
from .preparation import corpus_snapshot, load_frozen_snapshot
from .records import IndexConfig, VectorStoreError, context_ids, metadata_for, original_id, validate_context


def eligible(chunk: PolicyChunk, mode: RetrievalMode, as_of: date) -> bool:
    return (chunk.status in (APPROVED, DRAFT) if mode == "draft_review" else
            chunk.status == APPROVED and chunk.effective_date is not None and chunk.effective_date <= as_of)


def search_options(query, mode, top_k, reference_date) -> date:
    if (not isinstance(query, str) or not query.strip() or len(query) > 16000
            or mode not in ("approved_only", "draft_review")
            or type(top_k) is not int or not 1 <= top_k <= 100
            or (reference_date is not None and type(reference_date) is not date)):
        raise VectorStoreError("invalid query, mode, top_k or reference date")
    return reference_date if reference_date is not None else datetime.now(timezone.utc).date()


class ChunkCatalog:
    """Canonical inputs must come from the verified loader (or explicit test fixtures)."""
    def __init__(self, canonical, chunks, config: IndexConfig, fingerprint: str, *, root: Path | None = None):
        self.canonical, self.chunks = tuple(canonical), tuple(chunks)
        self.config, self.fingerprint, self.root = config, fingerprint, root
        originals = {c.chunk_id: c for c in self.canonical}
        if (not originals or len(originals) != len(self.canonical) or not self.chunks
                or not isinstance(fingerprint, str) or not fingerprint):
            raise VectorStoreError("catalog requires unique canonical chunks and corpus identity")
        try:
            validate_context(self.chunks, config)
            groups = defaultdict(list)
            stable_fields = {f.name for f in fields(PolicyChunk)} - {"chunk_id", "text", "source_start", "source_end"}
            for c in self.chunks:
                metadata_for(c, config.model)
                source = originals.get(original_id(c))
                if source is None or any(getattr(c, f) != getattr(source, f) for f in stable_fields):
                    raise ValueError("chunk metadata does not match its canonical source")
                if isinstance(c, RefinedPolicyChunk):
                    for s in c.segments:
                        if (any(type(v) is not int for v in (s.source_start, s.source_end, s.text_start, s.text_end))
                                or s.source_end <= s.source_start or s.text_end <= s.text_start):
                            raise ValueError("invalid refined spans")
                    primary = [s for s in c.segments if s.role == "source"]
                    if (not primary or c.source_start != min(s.source_start for s in primary)
                            or c.source_end != max(s.source_end for s in primary)):
                        raise ValueError("primary span bounds differ")
                groups[original_id(c)].append(c)
            if groups.keys() != originals.keys():
                raise ValueError("catalog lost canonical policy content")
            for key, source in originals.items():
                reconstruct(source, groups[key])
        except (ValueError, TypeError, AttributeError) as exc:
            raise VectorStoreError(f"catalog integrity failed: {exc}") from None
        self.by_id = MappingProxyType({c.chunk_id: c for c in self.chunks})

    def check_current(self, fingerprint: str) -> None:
        if fingerprint != self.fingerprint or (self.root is not None and corpus_snapshot(self.root)[1] != self.fingerprint):
            raise VectorStoreError("catalog is stale or belongs to another corpus")

    def siblings(self, chunk: PolicyChunk, mode: RetrievalMode, as_of: date) -> tuple[PolicyChunk, ...]:
        result = []
        for key in context_ids(chunk):
            c = self.by_id.get(key)
            if c is None or not eligible(c, mode, as_of):
                raise VectorStoreError("required sibling context missing or ineligible")
            result.append(c)
        return tuple(sorted(result, key=lambda c: (c.chunk_index, getattr(c, "refinement_index", 0), c.chunk_id)))

    def validate_hit(self, hit: RetrievalHit, mode: RetrievalMode, as_of: date) -> None:
        score_fields(hit)
        if (hit.model != self.config.model or self.by_id.get(hit.chunk.chunk_id) != hit.chunk
                or not eligible(hit.chunk, mode, as_of)):
            raise VectorStoreError("retrieval payload is ineligible or differs from the verified catalog")
        expected = {c.chunk_id: c for c in self.siblings(hit.chunk, mode, as_of)}
        supplied = {c.chunk_id: c for c in hit.context}
        if len(supplied) != len(hit.context) or supplied != expected:
            raise VectorStoreError("retrieved sibling context is missing, extra or differs from source")


def load_catalog(root: Path, snapshot_id: str) -> ChunkCatalog:
    """Verify saved chunk text against current sources without loading/encoding a model."""
    root = validate_project_root(root)
    if not isinstance(snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{64}", snapshot_id):
        raise VectorStoreError("an explicit immutable local snapshot ID is required")
    documents, fingerprint = corpus_snapshot(root)
    snapshot = load_frozen_snapshot(root, "dev-" + snapshot_id)
    if snapshot.corpus_fingerprint != fingerprint:
        raise VectorStoreError("saved snapshot does not match current corpus")
    canonical = tuple(c for d in documents for c in chunk_document(d).chunks)
    catalog = ChunkCatalog(canonical, (r.chunk for r in snapshot.records), snapshot.config, fingerprint, root=root)
    catalog.check_current(fingerprint)
    return catalog
