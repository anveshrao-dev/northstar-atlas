"""Prepare offline evidence, preflight a model, or explicitly opt in to inference."""
import argparse
from dataclasses import asdict
from datetime import date
import json
import os
from pathlib import Path
import sys

from northstar_atlas.generation import GenerationError, generate_answer, render_answer, retrieve_context
from northstar_atlas.llm import GenerationLimits, MODEL_ID
from northstar_atlas.nebius import NebiusProvider, ProviderError, load_settings
from northstar_atlas.qa import open_retriever
from northstar_atlas.project import ProjectRootError, validate_project_root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="Load credentials privately; GET /models only. No inference.")
    for name in ("prepare", "ask"):
        command = commands.add_parser(name)
        command.add_argument("question")
        command.add_argument("--backend", choices=("chroma", "pinecone"), default=os.environ.get("VECTOR_STORE", "chroma"))
        command.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
        command.add_argument("--reference-date", type=date.fromisoformat)
        command.add_argument("--context-tokens", type=int, default=32768)
        command.add_argument("--output-tokens", type=int, default=1024)
        if name == "ask": command.add_argument("--allow-paid", action="store_true", help="Use only after separate inference approval.")
    args = parser.parse_args(argv)
    try:
        root = validate_project_root(Path.cwd())
        if args.command == "ask" and not args.allow_paid:
            raise GenerationError("paid inference disabled; obtain approval before using --allow-paid")
        if args.command == "preflight":
            with NebiusProvider(load_settings(root)) as provider:
                result = provider.preflight()
                print(json.dumps(result, indent=2))
                return 0 if result["model_listed"] else 1
        settings = load_settings(root) if args.command == "ask" else None
        limits = GenerationLimits(args.context_tokens, args.output_tokens)
        with open_retriever(root, args.backend) as (store, embedder, fingerprint):
            prepared = retrieve_context(args.question, store, embedder, corpus_fingerprint=fingerprint,
                        mode=args.mode, reference_date=args.reference_date, limits=limits,
                        model=settings.model if settings else MODEL_ID)
        if args.command == "prepare":
            print(json.dumps({"backend": args.backend, "mode": args.mode,
                  "reference_date": prepared.context.reference_date.isoformat(), "model": prepared.request.model,
                  "corpus_fingerprint": fingerprint, "prompt_version": prepared.prompt_version,
                  "schema_version": prepared.schema_version, "validation_version": prepared.validation_version,
                  "reviewed_obligation_count": len(prepared.obligations), "token_counter": prepared.context.token_counter,
                  "input_token_bound": prepared.request.input_token_bound, "output_tokens": limits.output_tokens,
                  "context_tokens": limits.context_tokens, "groups": [asdict(g.citation) for g in prepared.context.groups],
                  "warnings": [asdict(w) for w in prepared.context.warnings]}, indent=2))
        else:
            with NebiusProvider(settings, allow_paid=True, max_attempts=1) as provider:
                result = generate_answer(prepared, provider)
            print(render_answer(result.answer))
            print(json.dumps({"model": prepared.request.model, "backend": args.backend,
                  "mode": args.mode, "reference_date": prepared.context.reference_date.isoformat(),
                  "corpus_fingerprint": fingerprint, "prompt_version": prepared.prompt_version,
                  "schema_version": prepared.schema_version, "validation_version": prepared.validation_version,
                  "condition_coverage": asdict(result.answer.coverage), "input_token_bound": prepared.request.input_token_bound,
                  "usage": asdict(result.response.usage) if result.response and result.response.usage else None,
                  "attempts": result.response.attempts if result.response else 0,
                  "latency_seconds": result.response.latency_seconds if result.response else 0,
                  "warnings": [asdict(w) for w in prepared.context.warnings]}, indent=2))
        return 0
    except (GenerationError, ProviderError, ProjectRootError) as exc:
        print(f"Generation stopped: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Retrieval/SDK/local errors may contain paths, keys or raw server bodies.
        print("Generation stopped: configuration or retrieval failure; details suppressed.", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
