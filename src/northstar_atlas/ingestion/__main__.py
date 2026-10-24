"""Inspect verified policy summaries: python -m northstar_atlas.ingestion."""

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import re
import sys

from northstar_atlas.ingestion.document_loader import DocumentLoadError, load_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and inspect local policy documents.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[3],
        help="Repository boundary (defaults to this editable source checkout).",
    )
    parser.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
    parser.add_argument("--reference-date", help="Inclusive effective-date cutoff, YYYY-MM-DD; default: today UTC.")
    args = parser.parse_args()
    as_of = datetime.now(timezone.utc).date()
    if args.reference_date is not None:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.reference_date):
                raise ValueError
            as_of = date.fromisoformat(args.reference_date)
        except ValueError:
            parser.error("--reference-date must be a valid date in YYYY-MM-DD format")
    try:
        documents = load_documents(args.project_root, mode=args.mode, reference_date=as_of)
    except DocumentLoadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Loaded {len(documents)} document(s); mode={args.mode}; reference_date={as_of}")
    print("ID\tTitle\tDepartment\tVersion\tStatus\tCharacters\tSHA-256")
    for doc in documents:
        print(f"{doc.document_id}\t{doc.title}\t{doc.department}\t{doc.version}\t"
              f"{doc.status}\t{len(doc.content)}\tPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
