"""Offline index/status/search inspection; full source text is opt-in."""

import argparse
from datetime import date
import json
from pathlib import Path
import sys

from northstar_atlas.embeddings import EmbeddingModel
from northstar_atlas.local_embeddings import MiniLMProvider, MODEL_ID, MODEL_REVISION
from . import ChromaStore, IndexConfig, corpus_snapshot, index_corpus, search_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Verify and index all policies, including labelled drafts.")
    status = commands.add_parser("status", help="Check persistence, integrity and source freshness without loading MiniLM.")
    search = commands.add_parser("search", help="Return similar chunks and required context; no generated answers.")
    for command in (index, status, search):
        command.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    search.add_argument("question")
    search.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--reference-date", type=date.fromisoformat)
    search.add_argument("--show-text", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "status":
            model = EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, "l2")
            state = ChromaStore(args.project_root, IndexConfig(model, 256)).status()
            _, fingerprint = corpus_snapshot(args.project_root)
            if state["state"] == "ready" and state["corpus_fingerprint"] != fingerprint:
                state.update(state="stale", reason="sources changed; rerun index")
            print(json.dumps(state, indent=2, ensure_ascii=False))
            return 0 if state["state"] == "ready" else 1
        # Validate the corpus before model loading or any database mutation.
        corpus_snapshot(args.project_root)
        provider = MiniLMProvider(args.project_root)
        if args.command == "index":
            result = index_corpus(args.project_root, provider)
            result["model_load_seconds"] = round(provider.load_seconds, 3)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            hits = search_corpus(args.project_root, args.question, provider, mode=args.mode,
                                top_k=args.top_k, reference_date=args.reference_date)
            print(f"Mode: {args.mode}; ranked hits: {len(hits)}")
            print("Cosine distance: lower is closer. Similarity = 1 - distance. Neither is answer confidence.")
            for rank, hit in enumerate(hits, 1):
                chunk = hit.chunk
                print(f"{rank}. {chunk.citation} | {chunk.status} | distance={hit.cosine_distance:.6f} "
                      f"similarity={hit.cosine_similarity:.6f}\n   ID: {chunk.chunk_id}\n   Source: {chunk.source_path}")
                for context in hit.context:
                    print(f"   Required context: {context.chunk_id} | {context.citation} (unranked)")
                if args.show_text:
                    for part in (chunk, *hit.context):
                        print(f"\n[{part.chunk_id}]\n{part.text}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Vector store error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
