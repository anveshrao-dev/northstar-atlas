"""Explicit approved Pinecone operations. No command implicitly creates an index."""

from northstar_atlas.project import ProjectRootError, validate_project_root

import argparse
from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
import sys
from time import perf_counter

from northstar_atlas.local_embeddings import MiniLMProvider
from northstar_atlas.vector_store.pinecone_cloud import (
    CloudError, check_confirmation, create_index, load_settings, preflight, sdk_client,
)
from northstar_atlas.vector_store.pinecone_state import PublicationState
from northstar_atlas.vector_store.pinecone_workflow import (
    RequestMeter, SMOKE_QUESTIONS, active_snapshot, connect_target, open_store,
    prepare_upload, save_report, search_report, status_report,
)
from northstar_atlas.vector_store.records import VectorStoreError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "create", "index", "status", "search", "smoke"):
        command = commands.add_parser(name)
        if name in ("index", "status"):
            command.add_argument("--namespace", help="Resume the exact saved immutable snapshot; no cleanup.")
        if name == "search":
            command.add_argument("question")
            command.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
            command.add_argument("--top-k", type=int, default=5)
        if name in ("search", "smoke"):
            command.add_argument("--reference-date", type=date.fromisoformat)
    args = parser.parse_args(argv)
    started = perf_counter()
    try:
        root = validate_project_root(Path.cwd())
        if getattr(args, "namespace", None) is not None:
            PublicationState.validate_namespace(args.namespace)
        settings = load_settings(root)
        check_confirmation(root, settings)
        meter = RequestMeter()
        with sdk_client(settings) as client:
            meter.attach(client)
            if args.command == "preflight":
                result = preflight(client.indexes, root, settings)
            elif args.command == "create":
                description = create_index(client.indexes, root, settings)
                result = {"state": "ready", "description": description}
            elif args.command == "status":
                result = status_report(root, settings, client, meter, namespace=args.namespace)
            else:
                target, index = connect_target(client, meter, settings)
                with index:
                    preparation = {}
                    selected = getattr(args, "namespace", None)
                    if args.command == "index" and selected is None and not PublicationState(root, asdict(target)).active():
                        runs = PublicationState(root, asdict(target)).runs()
                        if runs: raise CloudError("previous attempt exists; inspect local manifests and resume with --namespace")
                        snapshot, preparation = prepare_upload(root)
                    else:
                        snapshot = active_snapshot(root, target, selected)
                    store = open_store(root, snapshot, target, client, index)
                    if args.command == "index":
                        result = {**asdict(store.synchronize()), **preparation, "state": store.status()["state"]}
                    else:
                        audit_started = perf_counter()
                        store.audit()
                        audit_seconds = perf_counter() - audit_started
                        provider = MiniLMProvider(root)
                        if args.command == "search":
                            result = search_report(store, args.question, provider, mode=args.mode, top_k=args.top_k,
                                                   reference_date=args.reference_date)
                        else:
                            result = {"searches": [{"test": code, "question": query,
                                       **search_report(store, query, provider, mode=mode,
                                                       reference_date=args.reference_date)}
                                      for code, mode, query in SMOKE_QUESTIONS]}
                        result.update(audit_seconds=round(audit_seconds, 3), model_load_seconds=round(provider.load_seconds, 3))
                    result.update(namespace=snapshot.namespace, corpus_fingerprint=snapshot.corpus_fingerprint,
                                  embedding_config=asdict(snapshot.config), target=asdict(target))
        result.update(command=args.command, total_seconds=round(perf_counter() - started, 3), usage=meter.report())
        path = save_report(root, args.command, result)
        print(json.dumps({**result, "report": str(path.relative_to(root))}, indent=2, ensure_ascii=False))
        return 1 if result.get("state") in ("incomplete", "not_created", "not_indexed", "initializing") else 0
    except (CloudError, VectorStoreError, ProjectRootError) as exc:
        print(f"Pinecone operation stopped: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # Unknown SDK/config/model errors may contain credentials or private paths.
        print("Pinecone operation stopped: unexpected failure; no provider details logged. Reconcile state before retrying.", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())
