"""Operator-authorized, one-shot semantic graph diagnostic. Never a UI entrypoint.

No generated invocation IDs, automatic workflow retries, pricing refresh,
amendment, ledger initialization or recovery. Accepted public content only.
"""
import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .diagnostic_window import digest, sha, read_private
from .live_budget import GRAPH, PersistentBudget, WorkflowRequest, _require, configured_budget
from .live_graph import open_live_graph_service
from .qa import PublicAnswer, QAFailure, QASuccess


@dataclass(frozen=True)
class DiagnosticOutcome:
    accepted: bool
    kind: str
    code: str
    public: PublicAnswer | None = field(default=None, repr=False)


def run_approval(plan_sha256: str) -> str:
    """A distinct operator approval from permission to amend the ledger."""
    digest(plan_sha256)
    return sha(('diagnostic-run:' + plan_sha256).encode('ascii'))


def run_diagnostic(budget: PersistentBudget, request: WorkflowRequest, *, plan_sha256: str,
                   approved_run_sha256: str, question_sha256: str,
                   allow_live: bool, other_writers_stopped: bool) -> DiagnosticOutcome:
    """One explicit invocation. Return only after durable settlement succeeds.

All identity, question, corpus, mode and date checks are repeated under the
authoritative lock. Provider construction and access I/O happen after reserve.
The window remains installed, including after a no-call preflight failure.
"""
    try:
        _require(allow_live is True and other_writers_stopped is True
                 and isinstance(budget, PersistentBudget) and isinstance(request, WorkflowRequest)
                 and request.profile == GRAPH and request.backend == 'chroma')
        _require(approved_run_sha256 == run_approval(plan_sha256))
        digest(question_sha256)
        _require(sha(request.question.encode('utf-8')) == question_sha256)
        with budget.reserve(request, diagnostic_approval=plan_sha256) as reservation:
            with open_live_graph_service(budget.root, backend='chroma', profile=GRAPH,
                                         budget=budget, reservation=reservation) as service:
                result = service.ask(request.question, mode=request.mode,
                    reference_date=request.reference_date, reservation=reservation)
            _require(isinstance(result, (QASuccess, QAFailure)))
        if isinstance(result, QASuccess):
            return DiagnosticOutcome(True, result.public.kind, 'accepted', result.public)
        return DiagnosticOutcome(False, 'error', 'workflow_rejected')
    except Exception:
        # Provider diagnostics are fixed, bounded records written by the stage
        # permit. Never include exception strings or rejected candidates here.
        return DiagnosticOutcome(False, 'error', 'diagnostic_denied_or_failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approve-run', required=True)
    parser.add_argument('--invocation-id', required=True)
    parser.add_argument('--question-file', type=Path, required=True)
    parser.add_argument('--question-sha256', required=True)
    parser.add_argument('--mode', choices=('approved_only', 'draft_review'), required=True)
    parser.add_argument('--backend', choices=('chroma',), required=True)
    parser.add_argument('--profile', choices=(GRAPH,), required=True)
    parser.add_argument('--reference-date', required=True)
    parser.add_argument('--allow-live', action='store_true', required=True)
    parser.add_argument('--other-writers-stopped', action='store_true', required=True,
                        help='All other live entrypoints stopped, automatic restarts disabled')
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(args.root)
        _require(args.question_file.is_relative_to(budget.root/'.local'))
        raw = read_private(budget, args.question_file, root=budget.root, limit=64000)
        digest(args.question_sha256)
        _require(len(raw) <= 64000 and sha(raw) == args.question_sha256)
        # Preserve exact UTF-8 bytes; even a trailing newline belongs to approval.
        request = WorkflowRequest(args.invocation_id, args.profile, raw.decode('utf-8'), args.mode,
                                  args.backend, date.fromisoformat(args.reference_date))
        outcome = run_diagnostic(budget, request, plan_sha256=args.plan,
            approved_run_sha256=args.approve_run, question_sha256=args.question_sha256,
            allow_live=args.allow_live, other_writers_stopped=args.other_writers_stopped)
        print(f'Diagnostic result: {outcome.kind}; {outcome.code}. No further workflow attempted.')
        return 0 if outcome.accepted else 1
    except Exception:
        print('Diagnostic denied or failed. No automatic retry; review private accounting.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
