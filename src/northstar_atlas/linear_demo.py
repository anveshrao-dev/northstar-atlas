"""Guarded 15-to-16 amendment and one Pinecone Standard RAG demonstration.

No initialization, refund, rollback, pricing refresh or automatic retry.
All commands require separate operator authorization; import performs no I/O.
"""
import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict
from datetime import date
from decimal import Decimal
import os
from pathlib import Path

from .budget_amendment import AmendmentPreview
from .budget_transition import _encode, _decode, _read_private, _write_private, _private_directory, _sync
from .diagnostic_retest import DiagnosticOutcome
from .diagnostic_window import WINDOW_NAME, digest, identity, sha, validate_contract, validate_plan, read_private
from .live_budget import (BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest,
                          GRAPH, LINEAR, _estimate, _require, _SLOTS, configured_budget)
from .llm import MODEL_ID
from .project import contained_path


def request_contract(budget, request):
    value = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode('utf-8')),
        mode=request.mode, backend=request.backend, profile=request.profile,
        reference_date=request.reference_date.isoformat(), model=MODEL_ID,
        corpus_sha256=sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))
    validate_contract(value, profile=LINEAR)
    return value


@contextmanager
def _operation(budget, previous, writers_stopped):
    try:
        _require(writers_stopped is True and budget.policy == BudgetPolicy(15))
        digest(previous)
        with budget._locked(diagnostic_approval=previous) as plan:
            _require(plan['kind'] == 'v2-diagnostic-amendment')
            yield plan
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('demonstration amendment denied or uncertain; keep writers stopped') from None


def _candidate(budget, source, previous, price, contract):
    validate_contract(contract, profile=LINEAR)
    _require(contract['corpus_sha256'] == sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))
    old, _, state = budget._validate_document(_decode(source), current=False)
    _require(state.version == 2 and state.attempts_consumed == 15
             and old['max_attempts'] == 15 and old['max_reserved_usd'] == '0.100000'
             and old.get('diagnostic_request_sha256') == sha(_encode(previous['request']))
             and len(old['workflows']) == 2
             and all(f['profile'] == GRAPH and f['status'] == 'uncertain' for f in old['workflows'])
             and old['workflows'][-1]['id'] == previous['request']['invocation_id']
             and old['legacy'] is not None and len(old['legacy']['attempts']) == 1
             and old['legacy']['attempts'][0]['status'] != 'reserved'
             and all(f['id'] != contract['invocation_id'] for f in old['workflows']))
    _require(isinstance(price, dict) and price.get('id') not in {p['id'] for p in old['pricing_revisions']})
    candidate = deepcopy(old)
    candidate.update(max_attempts=16, diagnostic_request_sha256=sha(_encode(contract)), active_pricing=price['id'])
    candidate['pricing_revisions'].append(deepcopy(price))
    validator = copy(budget); validator.policy = BudgetPolicy(16)
    _, prices, after = validator._validate_document(candidate)
    _require(after == state)
    cost = sum((_estimate(prices[price['id']], s[2], s[3]) for s in _SLOTS[LINEAR]), Decimal(0))
    _require(state.reserved_usd + cost <= budget.policy.max_reserved_usd)
    return candidate, state, cost, validator


def _plan(budget, source, candidate, contract, previous_digest):
    price = candidate['pricing_revisions'][-1]
    return dict(version=2, kind='v2-linear-demo-amendment', authority=str(budget.authority_root),
        directory_identity=identity(budget.directory), lock_identity=identity(budget.lock_path),
        ledger_identity=identity(budget.path), source_sha256=sha(source), candidate_sha256=sha(_encode(candidate)),
        scope=budget.scope, billing_scope=budget.billing_scope, model=MODEL_ID,
        old_cap=15, new_cap=16, cost_cap='0.100000', pricing_sha256=sha(_encode(price)),
        request=contract, previous_window_sha256=previous_digest)


def prepare_demo(budget, contract, *, previous_plan_sha256, confirmed_billing_scope,
                 approved_max_attempts, approved_max_reserved_usd, writers_stopped,
                 pricing_revision, verified_pricing_sha256, dry_run=True):
    """Read-only by default; append a separately verified fresh pricing revision."""
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        _require(confirmed_billing_scope == budget.billing_scope and type(approved_max_attempts) is int
                 and approved_max_attempts == 16 and approved_max_reserved_usd == '0.100000'
                 and type(dry_run) is bool)
        digest(verified_pricing_sha256)
        _require(sha(_encode(pricing_revision)) == verified_pricing_sha256)
        source = budget._read_bytes()
        window = _read_private(budget, budget.directory/WINDOW_NAME)
        _require(sha(window) == previous_plan_sha256)
        candidate, state, cost, _ = _candidate(budget, source, previous, pricing_revision, contract)
        plan = _plan(budget, source, candidate, contract, previous_plan_sha256)
        raw = _encode(plan); plan_id = sha(raw)
        directory = None
        if not dry_run:
            directory = budget.directory/('linear-demo-' + plan_id)
            directory.mkdir(mode=0o700); _sync(budget.directory)
            for name, data in (('backup.v2.json', source), ('backup.window.json', window),
                               ('candidate.v2.json', _encode(candidate)), ('plan.json', raw)):
                _write_private(budget, directory/name, data)
        return AmendmentPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                state.attempts_consumed, str(state.reserved_usd), str(cost), directory)


def apply_demo(budget, plan_id, *, approved_plan_sha256, previous_plan_sha256, writers_stopped):
    """Replace only the approved candidate and window under the existing lock/guard."""
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        digest(plan_id); digest(approved_plan_sha256)
        _require(plan_id == approved_plan_sha256)
        directory = budget.directory/('linear-demo-' + plan_id)
        _private_directory(budget, directory)
        raw = _read_private(budget, directory/'plan.json')
        _require(len(raw) <= 16384 and sha(raw) == plan_id)
        plan = _decode(raw); validate_plan(budget, plan)
        _require(plan['kind'] == 'v2-linear-demo-amendment' and plan['previous_window_sha256'] == previous_plan_sha256
                 and not os.path.lexists(directory/'apply.started'))
        source = _read_private(budget, directory/'backup.v2.json')
        window = _read_private(budget, directory/'backup.window.json')
        candidate_raw = _read_private(budget, directory/'candidate.v2.json')
        _require(sha(source) == plan['source_sha256'] and sha(candidate_raw) == plan['candidate_sha256']
                 and sha(window) == previous_plan_sha256 and budget._read_bytes() == source
                 and identity(budget.path) == plan['ledger_identity']
                 and _read_private(budget, budget.directory/WINDOW_NAME) == window)
        candidate = _decode(candidate_raw)
        expected, state, _, validator = _candidate(budget, source, previous, candidate['pricing_revisions'][-1], plan['request'])
        _require(candidate_raw == _encode(expected)
                 and plan == _plan(budget, source, expected, plan['request'], previous_plan_sha256))

        def before_replace():
            _require(budget._read_bytes() == source and identity(budget.path) == plan['ledger_identity']
                     and _read_private(budget, budget.directory/WINDOW_NAME) == window)
            _write_private(budget, directory/'apply.started', _encode({'plan_sha256': plan_id}))
            next_window = directory/'window.next.json'
            _write_private(budget, next_window, raw)
            os.replace(next_window, budget.directory/WINDOW_NAME)
            _sync(directory); _sync(budget.directory)

        def verify():
            _require(budget._read_bytes() == candidate_raw and validator._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw)
            _write_private(budget, directory/'apply.completed', _encode({'plan_sha256': plan_id}))

        budget._settle(candidate_raw, before_replace=before_replace, verify=verify)
        return plan_id


def run_approval(plan_sha256):
    digest(plan_sha256)
    return sha(('linear-demo-run:' + plan_sha256).encode('ascii'))


def run_demo(budget, request, *, plan_sha256, approved_run_sha256, question_sha256,
             allow_live, other_writers_stopped):
    """One fixed LINEAR invocation. No retry, ID replacement or window removal."""
    from .live_preflight import require_private_preview
    from .qa import open_qa_service, QASuccess, QAFailure
    try:
        _require(allow_live is True and other_writers_stopped is True
                 and os.environ.get('NORTHSTAR_CONTROLLED_LIVE') == '1'
                 and isinstance(budget, PersistentBudget) and budget.policy == BudgetPolicy(16)
                 and budget.before_request is require_private_preview and isinstance(request, WorkflowRequest)
                 and request.profile == LINEAR and request.backend == 'pinecone')
        digest(question_sha256)
        _require(approved_run_sha256 == run_approval(plan_sha256)
                 and sha(request.question.encode('utf-8')) == question_sha256)
        def execute_once():
            with budget.reserve(request, diagnostic_approval=plan_sha256) as reservation:
                reservation.check_setup(budget, profile=LINEAR, backend='pinecone')
                _require(reservation.envelope == (1, 30720, 2048))
                require_private_preview()
                reservation.check_setup(budget, profile=LINEAR, backend='pinecone')
                # Existing opener verifies Pinecone confirmation, publication and remote audit.
                # It constructs a reservation-required, one-attempt Nebius adapter.
                with open_qa_service(budget.root, backend='pinecone', allow_paid=True, model=MODEL_ID,
                                     budget=budget, reservation=reservation) as service:
                    reservation.check_setup(budget, profile=LINEAR, backend='pinecone')
                    result = service.ask(request.question, mode=request.mode,
                                         reference_date=request.reference_date, reservation=reservation)
                _require(isinstance(result, (QASuccess, QAFailure)))
            return result

        from .result_capture import capture_result
        result = capture_result(execute_once, root=budget.root, invocation_id=request.invocation_id)
        if isinstance(result, QASuccess):
            return DiagnosticOutcome(True, result.public.kind, 'accepted', result.public)
        return DiagnosticOutcome(False, 'error', 'workflow_rejected')
    except Exception:
        return DiagnosticOutcome(False, 'error', 'demonstration_denied_or_failed')


def format_result(result: DiagnosticOutcome) -> str:
    """Present only an outcome released after accounting and private capture."""
    summary = f'Demonstration result: {result.kind}; {result.code}. No further invocation attempted.'
    if result.accepted and result.public is not None:
        return result.public.text + '\n' + summary
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--contract', type=Path, required=True)
    prepare.add_argument('--confirmed-billing-scope', required=True)
    prepare.add_argument('--approve-max-attempts', type=int, required=True)
    prepare.add_argument('--approve-max-reserved-usd', required=True)
    prepare.add_argument('--pricing-revision', type=Path, required=True)
    prepare.add_argument('--verified-pricing-sha256', required=True)
    prepare.add_argument('--write-artifacts', action='store_true')
    apply = sub.add_parser('apply')
    apply.add_argument('--plan', required=True); apply.add_argument('--approve', required=True)
    for command in (prepare, apply):
        command.add_argument('--previous-plan', required=True)
        command.add_argument('--writers-stopped', action='store_true', required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan', required=True); run.add_argument('--approve-run', required=True)
    run.add_argument('--invocation-id', required=True)
    run.add_argument('--question-file', type=Path, required=True)
    run.add_argument('--question-sha256', required=True)
    run.add_argument('--mode', choices=('approved_only', 'draft_review'), required=True)
    run.add_argument('--backend', choices=('pinecone',), required=True)
    run.add_argument('--profile', choices=(LINEAR,), required=True)
    run.add_argument('--reference-date', required=True)
    run.add_argument('--allow-live', action='store_true', required=True)
    run.add_argument('--other-writers-stopped', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(args.root)
        def private(path):
            _require(path.is_relative_to(budget.root/'.local'))
            return _read_private(budget, path, root=budget.root)
        if args.command == 'prepare':
            preview = prepare_demo(budget, _decode(private(args.contract)),
                previous_plan_sha256=args.previous_plan, confirmed_billing_scope=args.confirmed_billing_scope,
                approved_max_attempts=args.approve_max_attempts, approved_max_reserved_usd=args.approve_max_reserved_usd,
                writers_stopped=args.writers_stopped, pricing_revision=_decode(private(args.pricing_revision)),
                verified_pricing_sha256=args.verified_pricing_sha256, dry_run=not args.write_artifacts)
            print(_encode({k: v for k, v in asdict(preview).items() if k != 'artifact_directory'}).decode())
        elif args.command == 'apply':
            apply_demo(budget, args.plan, approved_plan_sha256=args.approve,
                       previous_plan_sha256=args.previous_plan, writers_stopped=args.writers_stopped)
            print('Demonstration amendment applied; ordinary entrypoints remain blocked.')
        else:
            _require(args.question_file.is_relative_to(budget.root/'.local'))
            raw = read_private(budget, args.question_file, root=budget.root, limit=64000)
            digest(args.question_sha256); _require(sha(raw) == args.question_sha256)
            request = WorkflowRequest(args.invocation_id, args.profile, raw.decode('utf-8'), args.mode,
                                      args.backend, date.fromisoformat(args.reference_date))
            result = run_demo(budget, request, plan_sha256=args.plan, approved_run_sha256=args.approve_run,
                question_sha256=args.question_sha256, allow_live=args.allow_live,
                other_writers_stopped=args.other_writers_stopped)
            print(format_result(result))
            return 0 if result.accepted else 1
        return 0
    except Exception:
        print('Demonstration denied or uncertain. No retry; review private accounting.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
