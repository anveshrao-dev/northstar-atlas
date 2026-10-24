"""Dense retrieval baseline; source coverage is not semantic adjudication.

Run: python -m northstar_atlas.evaluation.retrieval --reference-date 2026-10-22
Only question text and the specified mode enter the existing retriever. No index
mutation, document encoding, answer generation, or external service is invoked.
"""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
from statistics import mean
import sys
from time import perf_counter

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.ingestion.chunker import PolicyChunk, chunk_document
from northstar_atlas.ingestion.document_loader import APPROVED, DRAFT
from northstar_atlas.ingestion.refiner import RefinedPolicyChunk, refine_chunks
from northstar_atlas.local_embeddings import MiniLMProvider, MODEL_ID, MODEL_REVISION
from northstar_atlas.vector_store import ChromaStore, IndexConfig, SearchHit, corpus_snapshot, search_corpus
from northstar_atlas.vector_store.records import context_ids, metadata_for
from northstar_atlas.vector_store.contracts import RetrievalHit, score_fields


K = 5
SCORING_VERSION = "complete-source-section-v1"
CONTRACT = {"A": ("approved_only", "answer"), "B": ("draft_review", "answer"),
            "C": ("draft_review", "clarify"), "D": ("draft_review", "abstain")}
METRICS = ("recall_at_5", "hit_at_5", "full_evidence_coverage_at_5", "reciprocal_rank_at_5",
           "section_id_recall_at_5")
Reference = tuple[str, str]


def _reference(value: dict) -> Reference:
    if not isinstance(value, dict) or set(value) != {"document_id", "section_id"}:
        raise ValueError("evidence must contain document_id and section_id")
    doc, section = value["document_id"], value["section_id"]
    if (not isinstance(doc, str) or not re.fullmatch(r"[A-Z]+-\d{3}", doc)
            or not isinstance(section, str) or not re.fullmatch(r"\d+(?:\.\d+)*", section)):
        raise ValueError("invalid document-qualified section reference")
    return doc, section


def _ref_dict(ref: Reference) -> dict[str, str]:
    return {"document_id": ref[0], "section_id": ref[1]}


def _matches(chunk: PolicyChunk, ref: Reference) -> bool:
    # Dot-delimited ancestry: 4 includes 4.1, but not 40.1; never cross documents.
    return chunk.document_id == ref[0] and (chunk.section_id == ref[1] or chunk.section_id.startswith(ref[1] + "."))


def _spans(chunk: PolicyChunk) -> list[tuple[int, int]]:
    if isinstance(chunk, RefinedPolicyChunk):
        # Duplicated context is real retrieved text and can contribute coverage.
        return [(s.source_start, s.source_end) for s in chunk.segments]
    return [(chunk.source_start, chunk.source_end)]


def _covers(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    cursor = start
    for left, right in sorted(spans):
        if right <= cursor:
            continue
        if left > cursor:
            break
        cursor = max(cursor, right)
        if cursor >= end:
            return True
    return False


class EvidenceCatalog:
    """Expected source spans and verified chunk representations, never rescue text.

    Matching requires the whole canonical section (all descendants for a parent).
    The evaluator compares offsets only; unretrieved source content is never added
    to a question's context. This conservative proxy can under-credit a useful
    partial section. Human semantic support judgments remain a separate activity.
    """

    def __init__(self, canonical: Sequence[PolicyChunk], prepared: Sequence[PolicyChunk] | None = None):
        self.canonical = tuple(canonical)
        chunks = tuple(prepared) if prepared is not None else self.canonical
        self.known = {c.chunk_id: c for c in chunks}
        if not self.canonical or len(self.known) != len(chunks):
            raise ValueError("catalog must be nonempty with unique prepared chunk IDs")

    def targets(self, ref: Reference) -> tuple[PolicyChunk, ...]:
        return tuple(c for c in self.canonical if _matches(c, ref))

    def covered(self, ref: Reference, chunks: Sequence[PolicyChunk]) -> bool:
        targets = self.targets(ref)
        spans = [span for c in chunks if _matches(c, ref) for span in _spans(c)]
        return bool(targets) and all(_covers(c.source_start, c.source_end, spans) for c in targets)


def _eligible(chunk: PolicyChunk, mode: str, as_of: date) -> bool:
    if mode == "draft_review":
        return chunk.status in (APPROVED, DRAFT)
    return chunk.status == APPROVED and chunk.effective_date is not None and chunk.effective_date <= as_of


def _chunk_record(chunk: PolicyChunk, model: EmbeddingModel) -> dict:
    return {"chunk_id": chunk.chunk_id, "text": chunk.text, **metadata_for(chunk, model)}


def score_question(question: dict, hits: Sequence[RetrievalHit], catalog: EvidenceCatalog, *,
                   reference_date: date, latency_seconds: float) -> dict:
    """Pure deterministic scoring of at most five ranked hits and their siblings.

    Expanded reciprocal rank is the inverse of the earliest DIRECT rank whose
    prefix plus required siblings completes any required unit. It is a trigger
    prefix metric, not a fabricated rank for an unranked sibling.
    """
    if (question.get("category") not in CONTRACT or
            (question.get("retrieval_mode"), question.get("expected_behavior")) != CONTRACT[question["category"]]):
        raise ValueError("invalid category, retrieval mode, or expected behavior")
    if type(reference_date) is not date or not math.isfinite(latency_seconds) or latency_seconds < 0:
        raise ValueError("invalid reference date or latency")
    raw_refs = question.get("required_evidence")
    if not isinstance(raw_refs, list) or not raw_refs:
        raise ValueError("required_evidence must be a nonempty list")
    required = [_reference(r) for r in raw_refs]
    if len(required) != len(set(required)) or any(not catalog.targets(r) for r in required):
        raise ValueError("duplicate or unknown required evidence")
    if question.get("acceptable_alternative_evidence", []):
        raise ValueError("this baseline does not score alternative evidence; review scoring before using new labels")
    mode = question["retrieval_mode"]
    direct: list[PolicyChunk] = []
    expanded: dict[str, PolicyChunk] = {}
    direct_prefix, expanded_prefix = [], []
    retrieved, sibling_records, violations = [], {}, []

    def valid(chunk: PolicyChunk, origin: str) -> bool:
        if not _eligible(chunk, mode, reference_date):
            violations.append({"chunk_id": chunk.chunk_id, "origin": origin, "reason": "authority_leakage"})
            return False
        if catalog.known.get(chunk.chunk_id) != chunk:
            violations.append({"chunk_id": chunk.chunk_id, "origin": origin, "reason": "invalid_provenance_or_text"})
            return False
        return True

    for rank, hit in enumerate(hits[:K], 1):
        scores = score_fields(hit)
        chunk = hit.chunk
        accepted = valid(chunk, "direct")
        retrieved.append({**_chunk_record(chunk, hit.model), "rank": rank, **scores,
                          "eligible_verified": accepted})
        if accepted:
            direct.append(chunk)
            expanded[chunk.chunk_id] = chunk
        required_ids = set(context_ids(chunk))
        supplied_ids = {c.chunk_id for c in hit.context}
        if required_ids - supplied_ids:
            violations.append({"chunk_id": chunk.chunk_id, "origin": "context", "reason": "missing_required_siblings",
                               "missing_ids": sorted(required_ids - supplied_ids)})
        for sibling in hit.context:
            sibling_ok = valid(sibling, "context")
            if sibling.chunk_id not in required_ids:
                violations.append({"chunk_id": sibling.chunk_id, "origin": "context", "reason": "unexpected_context"})
                sibling_ok = False
            sibling_ok = sibling_ok and accepted
            if sibling.chunk_id not in sibling_records:
                unranked = ({"cosine_distance": None, "cosine_similarity": None}
                            if isinstance(hit, SearchHit) else {"score_kind": None})
                sibling_records[sibling.chunk_id] = {**_chunk_record(sibling, hit.model), "rank": None,
                    **unranked, "trigger_ranks": [],
                    "valid_trigger_ranks": [], "occurrences": [], "eligible_verified": False}
            record = sibling_records[sibling.chunk_id]
            record["trigger_ranks"].append(rank)
            record["occurrences"].append({"trigger_rank": rank, "trigger_chunk_id": chunk.chunk_id,
                                           "eligible_verified": sibling_ok})
            if sibling_ok:
                # A previous invalid occurrence must not hide a later verified one.
                record.update(_chunk_record(sibling, hit.model))
                record["eligible_verified"] = True
                record["valid_trigger_ranks"].append(rank)
                expanded[sibling.chunk_id] = sibling
        direct_prefix.append([r for r in required if catalog.covered(r, direct)])
        expanded_prefix.append([r for r in required if catalog.covered(r, tuple(expanded.values()))])

    def metrics(chunks: Sequence[PolicyChunk], prefixes: list[list[Reference]]) -> dict:
        complete = prefixes[-1] if prefixes else []
        present = [r for r in required if any(_matches(c, r) for c in chunks)]
        first = next((i for i, values in enumerate(prefixes, 1) if values), None)
        return {"recall_at_5": len(complete) / len(required), "hit_at_5": int(bool(complete)),
                "full_evidence_coverage_at_5": int(len(complete) == len(required)),
                "reciprocal_rank_at_5": 1 / first if first else 0.0,
                "section_id_recall_at_5": len(present) / len(required),
                "covered_evidence": [_ref_dict(r) for r in complete],
                "missing_evidence": [_ref_dict(r) for r in required if r not in complete],
                "partial_evidence": [_ref_dict(r) for r in present if r not in complete]}

    return {"question_id": question["question_id"], "question": question["question"],
            "category": question["category"], "retrieval_mode": mode,
            "expected_behavior": question["expected_behavior"], "answer_behavior_evaluated": None,
            "evidence_purpose": {"answer": "answer_support", "clarify": "conditional_rules", "abstain": "corpus_limits"}[question["expected_behavior"]],
            "required_evidence": [_ref_dict(r) for r in required], "retrieved": retrieved,
            "sibling_context": list(sibling_records.values()), "violations": violations,
            "direct": metrics(direct, direct_prefix), "expanded": metrics(tuple(expanded.values()), expanded_prefix),
            "latency_seconds": round(latency_seconds, 6),
            "context_budget": {"ranked_slots": len(retrieved), "direct_unique_chunks": len({c.chunk_id for c in direct}),
                               "expanded_unique_chunks": len(expanded), "direct_characters": sum(len(c.text) for c in direct),
                               "expanded_characters": sum(len(c.text) for c in expanded.values())}}


def summarize(results: Sequence[dict]) -> dict:
    if not results:
        raise ValueError("cannot summarize an empty evaluation")

    def group(rows: Sequence[dict]) -> dict:
        return {"questions": len(rows),
                **{view: {name: mean(r[view][name] for r in rows) for name in METRICS} for view in ("direct", "expanded")},
                "average_latency_seconds": mean(r["latency_seconds"] for r in rows),
                "slowest": {"question_id": max(rows, key=lambda r: r["latency_seconds"])["question_id"],
                            "latency_seconds": max(r["latency_seconds"] for r in rows)},
                "authority_leakage": sum(v["reason"] == "authority_leakage" for r in rows for v in r["violations"]),
                "validation_violations": sum(len(r["violations"]) for r in rows)}

    categories = sorted({r["category"] for r in results})
    answerable = [r for r in results if r["expected_behavior"] == "answer"]
    return {"overall": group(results), "answerable": group(answerable) if answerable else None,
            "by_category": {c: group([r for r in results if r["category"] == c]) for c in categories}}


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError("evaluation path resolves outside project")
    return path


def load_dataset(root: Path) -> tuple[dict, str]:
    """Reuse the dataset validator without changing its behavior or sys.path."""
    root = Path(root).resolve(strict=True)
    path = _inside(root, "scripts/validate_evaluations.py")
    spec = importlib.util.spec_from_file_location("northstar_m18_validator", path)
    if spec is None or spec.loader is None:
        raise ValueError("M1.8 validator is unavailable")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    corpus, manifest_hash = validator.load_corpus(root)
    data, dataset_hash = validator.read_json(_inside(root, "evaluations/questions.json"))
    errors = validator.validate_dataset(data, corpus, manifest_hash)
    if errors:
        raise ValueError("invalid evaluation dataset: " + "; ".join(errors))
    if any(q.get("acceptable_alternative_evidence", []) for q in data["questions"]):
        raise ValueError("alternative evidence is not supported by this baseline scoring version")
    return data, dataset_hash


def format_results(results: dict) -> str:
    """Stable key order/UTF-8/numbers; run timestamps and timings naturally vary."""
    return json.dumps(results, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_results(root: Path, results: dict, *, run_name: str) -> Path:
    root = Path(root).resolve(strict=True)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run name must be a simple filename stem")
    directory = _inside(root, ".local/evaluations/m2.5")
    destination = _inside(root, f".local/evaluations/m2.5/{run_name}.json")
    serialized = format_results(results)
    directory.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return destination


def evaluate(root: Path, *, reference_date: date) -> dict:
    """Run all questions once, in dataset order, against an already-ready index."""
    root = Path(root).resolve(strict=True)
    dataset, dataset_hash = load_dataset(root)
    documents, fingerprint = corpus_snapshot(root)
    config = IndexConfig(EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, "l2"), 256)
    store = ChromaStore(root, config)
    initial_state = store.status()
    if initial_state.get("state") != "ready" or initial_state.get("corpus_fingerprint") != fingerprint:
        raise ValueError("index is not ready/compatible with this corpus; evaluation will not rebuild it")
    # Read Chroma's effective search configuration; do not change collection settings.
    with store._lock():
        collection_configuration = store._collection().configuration_json
    setup_started = perf_counter()
    provider = MiniLMProvider(root)
    canonical = tuple(c for d in documents for c in chunk_document(d).chunks)
    prepared = refine_chunks(canonical, provider).require_complete()
    if initial_state["count"] != len(prepared):
        raise ValueError("index count differs from the current prepared corpus")
    catalog = EvidenceCatalog(canonical, prepared)
    setup_seconds = perf_counter() - setup_started
    results = []
    for question in dataset["questions"]:
        # Expected summaries, evidence IDs and qualifiers are never search inputs.
        started = perf_counter()
        hits = search_corpus(root, question["question"], provider, mode=question["retrieval_mode"],
                             top_k=K, reference_date=reference_date)
        elapsed = perf_counter() - started
        result = score_question(question, hits, catalog, reference_date=reference_date, latency_seconds=elapsed)
        direct_texts = [h.chunk.text for h in hits]
        expanded = {c.chunk_id: c.text for h in hits for c in (h.chunk, *h.context)}
        result["context_budget"].update(direct_tokens=sum(provider.token_counts(direct_texts)),
                                        expanded_tokens=sum(provider.token_counts(list(expanded.values()))))
        results.append(result)
    if (store.status() != initial_state or corpus_snapshot(root)[1] != fingerprint
            or hashlib.sha256(_inside(root, "evaluations/questions.json").read_bytes()).hexdigest() != dataset_hash):
        raise ValueError("index, corpus, or dataset changed during evaluation; discard this run")
    with store._lock():
        if store._collection().configuration_json != collection_configuration:
            raise ValueError("collection configuration changed during evaluation; discard this run")
    code_paths = ["scripts/validate_evaluations.py", "pyproject.toml", "uv.lock"]
    code_paths += sorted(p.relative_to(root).as_posix() for p in (root / "src/northstar_atlas").rglob("*.py"))
    provenance = {"dataset_id": dataset["dataset_id"], "dataset_version": dataset["dataset_version"],
        "dataset_sha256": dataset_hash, "manifest_sha256": dataset["corpus_manifest_sha256"],
        "corpus_fingerprint": fingerprint, "sources": [{"document_id": d.document_id, "version": d.version,
            "status": d.status, "effective_date": d.effective_date.isoformat() if d.effective_date else None,
            "source_path": d.source_path.as_posix(), "content_hash": d.content_hash} for d in documents],
        "index_state": initial_state, "index_configuration": asdict(config),
        "chroma_collection_configuration": collection_configuration,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("chromadb", "sentence-transformers", "torch", "transformers", "numpy", "tokenizers")},
        "code_sha256": {p: hashlib.sha256(_inside(root, p).read_bytes()).hexdigest() for p in code_paths}}
    return {"schema_version": 1, "scoring_version": SCORING_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "reference_date": reference_date.isoformat(),
        "top_k": K, "backend": "chroma_dense_cosine", "provenance": provenance,
        "scoring": {"basis": "conservative complete source-section coverage, not human-adjudicated semantic support",
            "parent_rule": "full own section and all dot-delimited descendants; same document only",
            "expanded_rr": "reciprocal direct trigger-prefix rank; siblings have no independent rank",
            "section_id_recall": "diagnostic only; partial source coverage earns no main-metric credit",
            "semantic_support_adjudicated": False, "answer_faithfulness_evaluated": False},
        "timing": {"setup_seconds": setup_seconds, "model_load_seconds": provider.load_seconds,
            "definition": "one search_corpus call: source/integrity validation, query embedding, Chroma query and sibling expansion; excludes setup, scoring, token accounting and output IO",
            "warmup": "model and index initialized; no benchmark-query warmup; one pass in dataset order"},
        "summary": summarize(results), "questions": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--reference-date", type=date.fromisoformat, required=True)
    parser.add_argument("--run-name", default=None, help="New result filename stem; existing files are never overwritten.")
    args = parser.parse_args()
    try:
        result = evaluate(args.project_root, reference_date=args.reference_date)
        run_name = args.run_name or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
        path = write_results(args.project_root, result, run_name=run_name)
        print(f"Results: {path}")
        print("Automated complete-section coverage; semantic support and answer faithfulness are not scored.")
        print(format_results(result["summary"]))
        return 0 if result["summary"]["overall"]["validation_violations"] == 0 else 1
    except (ValueError, OSError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
