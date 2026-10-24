"""Local dense/BM25/RRF comparison; no generation, uploads or automatic rebuilds.

Requires an existing verified local snapshot and ready Chroma index. Results are
new files under .local/evaluations/m5; historical evaluation files are untouched.
"""
import argparse
from dataclasses import asdict
from datetime import date, datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
from statistics import mean, median
import sys
from time import perf_counter

from northstar_atlas.context import build_context
from northstar_atlas.generation import GenerationError, prepare_answer
from northstar_atlas.llm import GenerationLimits, QwenTokenBound
from northstar_atlas.project import validate_project_root
from northstar_atlas.vector_store.bm25 import ALGORITHM_VERSION, TOKENIZER_VERSION, BM25Store
from northstar_atlas.vector_store.catalog import load_catalog
from northstar_atlas.vector_store.contracts import SearchHit, score_fields
from northstar_atlas.vector_store.hybrid import CANDIDATES, RRF_CONSTANT, fuse_results
from northstar_atlas.vector_store.preparation import load_frozen_snapshot
from northstar_atlas.vector_store.records import VectorStoreError, context_ids, digest
from .retrieval import EvidenceCatalog, SCORING_VERSION, format_results, load_dataset, score_question, summarize

ARMS = ("dense", "bm25", "hybrid")


def _candidate_records(hits):
    return [{"chunk_id": h.chunk.chunk_id, "rank": rank, "citation": h.chunk.citation,
             "document_id": h.chunk.document_id, "section_id": h.chunk.section_id,
             "version": h.chunk.version, "status": h.chunk.status, "content_hash": h.chunk.content_hash,
             "required_context_ids": list(context_ids(h.chunk)),
             **({"score_kind": "cosine_distance"} if isinstance(h, SearchHit) else {}), **score_fields(h)}
            for rank, h in enumerate(hits, 1)]


def aks_diagnostic() -> dict:
    return {"question_id": "SUPP-AKS-001", "category": "B", "retrieval_mode": "draft_review",
            "expected_behavior": "answer", "question": "An engineer needs temporary production access to troubleshoot an AKS issue. Who should approve it, and which team grants the access?",
            "required_evidence": [{"document_id": d, "section_id": s} for d, s in
                                  (("IT-001", "10.1"), ("SEC-001", "9.1"), ("SEC-001", "5.2"),
                                   ("IT-001", "9.1"), ("SEC-001", "6.1"))]}


def _coverage(question, chunks, catalog):
    required = [(r["document_id"], r["section_id"]) for r in question["required_evidence"]]
    covered = [r for r in required if catalog.covered(r, chunks)]
    return {"recall": len(covered) / len(required), "full_coverage": len(covered) == len(required),
            "missing_evidence": [{"document_id": d, "section_id": s} for d, s in required if (d,s) not in covered]}


def context_diagnostics(question, hits, catalog, *, reference_date: date,
                        limits: GenerationLimits = GenerationLimits()) -> dict:
    """Separate an assembled package from evidence actually eligible for generation."""
    started = perf_counter()
    package = build_context(hits, counter=QwenTokenBound(),
                            max_tokens=limits.context_tokens-limits.output_tokens-limits.margin_tokens,
                            mode=question["retrieval_mode"], reference_date=reference_date)
    input_bound, error = None, None
    try:
        prepared = prepare_answer(question["question"], tuple(hits), mode=question["retrieval_mode"],
                                  reference_date=reference_date, limits=limits)
        input_bound = prepared.request.input_token_bound
    except GenerationError as exc:
        error = str(exc)  # Local application validation only; no provider response exists.
    chunks = tuple(s.chunk for group in package.groups for s in group.sources)
    return {"generation_ready": error is None, "error": error,
            "warnings": [asdict(w) for w in package.warnings], "group_count": len(package.groups),
            "unique_chunks": len({c.chunk_id for c in chunks}),
            "citations": [g.citation.label for g in package.groups],
            "deduplicated_characters": sum(len(s.text) for g in package.groups for s in g.spans),
            "source_spans": [{"document_id": g.citation.document_id, "section_id": g.citation.section_id,
                               "content_hash": g.citation.content_hash, "start": s.source_start, "end": s.source_end}
                              for g in package.groups for s in g.spans],
            "context_token_bound": package.token_count, "counter": QwenTokenBound.name,
            "full_input_token_bound": input_bound, "limits": asdict(limits),
            "package_evidence": _coverage(question, chunks, catalog),
            "delivered_evidence": _coverage(question, chunks if error is None else (), catalog),
            "preparation_seconds": perf_counter()-started}


def evaluate_case(question, dense, lexical: BM25Store, shared, *, reference_date: date,
                  arm_order=ARMS) -> dict:
    """Expected evidence enters scoring only, never any retriever or query encoder."""
    if set(arm_order) != set(ARMS) or len(arm_order) != len(ARMS):
        raise ValueError("all three retrieval arms must run once")
    catalog = lexical.catalog
    scoring = EvidenceCatalog(catalog.canonical, catalog.chunks)
    options = dict(corpus_fingerprint=catalog.fingerprint, mode=question["retrieval_mode"], reference_date=reference_date)
    rows, stages, dense20 = {}, {}, ()

    def checked_search(store, k, provider):
        started = perf_counter()
        values = store.search(question["question"], provider, top_k=k, **options)
        if len(values) > k or len({h.chunk.chunk_id for h in values}) != len(values):
            raise VectorStoreError("invalid direct result count or duplicate IDs")
        for h in values:
            catalog.validate_hit(h, options["mode"], reference_date)
        return values, perf_counter()-started

    for arm in arm_order:
        if arm == "hybrid":
            dense20, dense_time = checked_search(dense, CANDIDATES, shared)
            lex20, lex_time = checked_search(lexical, CANDIDATES, None)
            start = perf_counter()
            hits = fuse_results(dense20, lex20, catalog, mode=options["mode"], reference_date=reference_date)
            fusion_time = perf_counter()-start
            elapsed = dense_time + lex_time + fusion_time
            stages[arm] = {"dense_seconds": dense_time, "bm25_seconds": lex_time, "fusion_seconds": fusion_time}
            stages["candidate_ranks"] = {"dense": [h.chunk.chunk_id for h in dense20],
                                          "bm25": [h.chunk.chunk_id for h in lex20]}
            stages["candidate_records"] = {"dense": _candidate_records(dense20),
                                            "bm25": _candidate_records(lex20)}
        else:
            hits, elapsed = checked_search(dense if arm == "dense" else lexical, 5,
                                           shared if arm == "dense" else None)
            stages[arm] = {"retrieval_seconds": elapsed}
        row = score_question(question, hits, scoring, reference_date=reference_date, latency_seconds=elapsed)
        if row["violations"]:
            raise VectorStoreError("retrieval evidence validation failed; comparison stopped")
        row["prepared_context"] = context_diagnostics(question, hits, scoring, reference_date=reference_date)
        rows[arm] = row
    control = score_question(question, dense20[:5], scoring, reference_date=reference_date, latency_seconds=0)
    return {"question_id": question["question_id"], "arm_order": list(arm_order), "arms": rows,
            "stages": stages, "dense20_prefix": control,
            "dense5_matches_dense20_prefix": [h["chunk_id"] for h in rows["dense"]["retrieved"]] ==
                                             [h.chunk.chunk_id for h in dense20[:5]]}


def _summary(values):
    return {"mean": mean(values), "median": median(values), "maximum": max(values)}


def _local(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError("evaluation path leaves project")
    return path


def write_results(root: Path, result: dict, *, run_name: str) -> Path:
    root = Path(root).resolve(strict=True)
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run name must be a simple filename stem")
    path = _local(root, f".local/evaluations/m5/{run_name}.json")
    text = format_results(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
    return path


def evaluate(root: Path, *, snapshot_id: str, reference_date: date, warm_passes: int = 3) -> dict:
    """Explicit local experiment; missing/stale state stops without rebuilding."""
    if type(warm_passes) is not int or not 0 <= warm_passes <= 3:
        raise ValueError("warm passes must be 0..3")
    root = validate_project_root(root)
    # Imports remain lazy so pure metric tests need no model/backend initialization.
    from northstar_atlas.local_embeddings import MiniLMProvider
    from .backends import ComparisonChromaStore
    from .comparison import SharedQueries

    started = perf_counter()
    dataset, dataset_hash = load_dataset(root)
    catalog = load_catalog(root, snapshot_id)
    snapshot = load_frozen_snapshot(root, "dev-"+snapshot_id)
    receipt_path = _local(root, ".local/chroma/index-state.json")
    receipt_bytes = receipt_path.read_bytes()
    dense = ComparisonChromaStore(root, snapshot, json.loads(receipt_bytes))
    before = dense.light_status()
    built = perf_counter()
    lexical = BM25Store(catalog)
    lexical_seconds = perf_counter()-built
    supplementary = [aks_diagnostic()]
    # Paraphrases require a separate human equivalence review; do not silently
    # promote design suggestions into new ground truth in this first experiment.
    provider = MiniLMProvider(root)
    shared = SharedQueries([*dataset["questions"], *supplementary], provider, catalog.config)
    setup_seconds = perf_counter()-started
    cases = [evaluate_case(q, dense, lexical, shared, reference_date=reference_date)
             for q in dataset["questions"]]
    diagnostics = [evaluate_case(q, dense, lexical, shared, reference_date=reference_date) for q in supplementary]
    timing_passes = []
    for pass_number in range(warm_passes):
        order = ARMS[pass_number:] + ARMS[:pass_number]
        rows = []
        for q, original in zip(dataset["questions"], cases, strict=True):
            result = evaluate_case(q, dense, lexical, shared, reference_date=reference_date, arm_order=order)
            for arm in ARMS:
                first_ids = [h["chunk_id"] for h in original["arms"][arm]["retrieved"]]
                if first_ids != [h["chunk_id"] for h in result["arms"][arm]["retrieved"]]:
                    raise ValueError("ranking changed between timing passes; investigate nondeterminism")
            rows.append({"question_id": q["question_id"], "stages": result["stages"],
                         "arms": {a: {"retrieval_seconds": r["latency_seconds"],
                                      "context_seconds": r["prepared_context"]["preparation_seconds"]}
                                  for a, r in result["arms"].items()}})
        timing_passes.append({"pass": pass_number+1, "order": list(order), "questions": rows})
    catalog.check_current(catalog.fingerprint)
    if receipt_path.read_bytes() != receipt_bytes or dense.light_status() != before or load_dataset(root)[1] != dataset_hash:
        raise ValueError("index, corpus or dataset changed during evaluation")
    summaries = {a: summarize([c["arms"][a] for c in cases]) for a in ARMS}
    context_summary = {a: {
        "expanded_chunks": _summary([c["arms"][a]["context_budget"]["expanded_unique_chunks"] for c in cases]),
        "expanded_characters": _summary([c["arms"][a]["context_budget"]["expanded_characters"] for c in cases]),
        "context_token_bound": _summary([c["arms"][a]["prepared_context"]["context_token_bound"] for c in cases]),
        "generation_ready_count": sum(c["arms"][a]["prepared_context"]["generation_ready"] for c in cases),
        "delivered_recall": mean(c["arms"][a]["prepared_context"]["delivered_evidence"]["recall"] for c in cases)
        } for a in ARMS}
    paths = ["pyproject.toml", "uv.lock", "scripts/validate_evaluations.py"]
    paths += sorted(p.relative_to(root).as_posix() for p in (root/"src/northstar_atlas").rglob("*.py"))
    return {"schema_version": 1, "scoring_version": SCORING_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "reference_date": reference_date.isoformat(),
            "parameters": {"bm25": {"k1": lexical.k1, "b": lexical.b, "algorithm": ALGORITHM_VERSION,
                                       "tokenizer": TOKENIZER_VERSION},
                           "rrf_constant": RRF_CONSTANT, "candidates_per_channel": CANDIDATES, "final_top_k": 5},
            "provenance": {"dataset_sha256": dataset_hash, "manifest_sha256": dataset["corpus_manifest_sha256"],
                           "snapshot_id": snapshot_id, "corpus_fingerprint": catalog.fingerprint,
                           "index_config": asdict(catalog.config), "record_count": len(catalog.chunks),
                           "chroma_receipt": before, "selected_records_verified": len(dense.checked_ids),
                           "maximum_vector_direction_error": dense.max_direction_error,
                           "query_vector_hashes": {q["question_id"]: digest(shared.embed_query(q["question"]))
                                                   for q in [*dataset["questions"], *supplementary]},
                           "python": platform.python_version(), "platform": platform.platform(),
                           "packages": {n: importlib.metadata.version(n) for n in
                                        ("chromadb","sentence-transformers","torch","transformers","numpy")},
                           "code_sha256": {p: hashlib.sha256(_local(root,p).read_bytes()).hexdigest() for p in paths}},
            "summary": summaries, "context_summary": context_summary, "questions": cases,
            "supplementary": diagnostics,
            "timing": {"setup_seconds": setup_seconds, "bm25_build_seconds": lexical_seconds,
                       "model_load_seconds": provider.load_seconds, "query_embedding_seconds": shared.timings,
                       "embedding_summary_seconds": _summary(list(shared.timings.values())),
                       "warm_retrieval_seconds": {a: _summary([q["arms"][a]["retrieval_seconds"]
                                                  for p in timing_passes for q in p["questions"]])
                                                  for a in ARMS} if timing_passes else {},
                       "passes": timing_passes,
                       "boundaries": "Retrieval includes readiness, selected payload checks and required siblings; excludes one-time query encoding, scoring, context preparation and artifact IO. Hybrid includes dense20+lexical20+fusion. Context timing includes offline full-prompt validation."},
            "limitations": ["No LLM inference or answer faithfulness evaluation.",
                            "Whole-section span coverage is not semantic adjudication.",
                            "No new Pinecone operations or reranking experiment.",
                            "Paraphrases deferred pending human equivalence review; exact AKS diagnostic is separate."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--reference-date", type=date.fromisoformat, default=date(2026,10,22))
    parser.add_argument("--warm-passes", type=int, default=3)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    try:
        root = validate_project_root(args.project_root)
        result = evaluate(root, snapshot_id=args.snapshot_id, reference_date=args.reference_date, warm_passes=args.warm_passes)
        name = args.run_name or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S%fZ")
        path = write_results(root, result, run_name=name)
        print(f"Results: {path.relative_to(root)}")
        print(format_results({a: result["summary"][a]["overall"] for a in ARMS}))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Local comparison stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
