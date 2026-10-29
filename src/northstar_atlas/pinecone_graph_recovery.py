"""Explicit device-identity recovery and one Pinecone semantic workflow.

Only existing v2 history at 15 accounted calls can become a 22-call authority.
No initialization, refunds, retries, rollback, implicit pricing or live defaults.
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
from .diagnostic_window import (WINDOW_NAME, RECOVERY_KIND, digest, sha, identity, validate_plan,
    _validate_plan_fields, validate_recovery_contract, verified_snapshot_binding)
from .live_budget import (BudgetError, BudgetPolicy, PersistentBudget, WorkflowRequest, GRAPH,
                         _fields, _require, _estimate, _SLOTS, configured_budget)
from .llm import MODEL_ID
from .project import contained_path


@contextmanager
def _operation(budget, writers_stopped, recover_identity):
    try:
        _require(writers_stopped is True and recover_identity is True
                 and isinstance(budget, PersistentBudget) and budget.policy == BudgetPolicy(15))
        with budget._physical_lock():
            yield
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('recovery denied or uncertain; keep writers stopped and review private evidence') from None


def _identities(budget):
    return dict(authority_identity=identity(budget.authority_root),
        local_identity=identity(budget.authority_root / '.local'),
        directory_identity=identity(budget.directory), lock_identity=identity(budget.lock_path),
        ledger_identity=identity(budget.path))


def _source(budget):
    """Read under physical exclusion; historical validity does not grant inference."""
    source = budget._read_bytes()
    raw_window = _read_private(budget, budget.directory / WINDOW_NAME)
    _require(len(raw_window) <= 16384)
    previous = _decode(raw_window)
    _validate_plan_fields(budget, previous)
    _require(previous['kind'] == 'v2-diagnostic-amendment')
    old, _, state = budget._validate_document(_decode(source), current=False)
    _require(state.version == 2 and state.attempts_consumed == 15
             and old['max_attempts'] == 15 and old['max_reserved_usd'] == '0.100000'
             and old.get('diagnostic_request_sha256') == sha(_encode(previous['request']))
             and len(old['workflows']) == 2
             and all(f['profile'] == GRAPH and f['status'] == 'uncertain' for f in old['workflows'])
             and old['workflows'][-1]['id'] == previous['request']['invocation_id']
             and old['workflows'][-1]['mode'] == previous['request']['mode']
             and old['workflows'][-1]['backend'] == previous['request']['backend']
             and old['legacy'] is not None and len(old['legacy']['attempts']) == 1
             and old['legacy']['attempts'][0]['status'] != 'reserved')
    # This utility addresses the reported device-only change, not replaced locks.
    # Equal inodes are an additional constraint, NEVER authorization by themselves.
    ids = _identities(budget)
    for key in ('directory_identity', 'lock_identity'):
        _require(previous[key][1] == ids[key][1] and previous[key][0] != ids[key][0])
    return source, raw_window, previous, old, state, ids


def inspect_recovery(budget, *, writers_stopped, recover_identity):
    """Read-only observations for independent review; no authorization or raw data."""
    with _operation(budget, writers_stopped, recover_identity):
        source, raw_window, _, _, state, ids = _source(budget)
        return dict(source_sha256=sha(source), previous_window_sha256=sha(raw_window),
                    identities_sha256=sha(_encode(ids)), attempts_preserved=state.attempts_consumed,
                    reserved_usd=str(state.reserved_usd))


def request_contract(budget, request: WorkflowRequest):
    value = dict(invocation_id=request.invocation_id, question_sha256=sha(request.question.encode('utf-8')),
        mode=request.mode, backend=request.backend, profile=request.profile,
        reference_date=request.reference_date.isoformat(), model=MODEL_ID,
        corpus_sha256=sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()),
        **verified_snapshot_binding(budget))
    validate_recovery_contract(value)
    return value


def _continuity(budget, evidence, approved, source, raw_window, ids):
    _fields(evidence, 'version authority scope billing_scope source_sha256 previous_window_sha256 '
                      'identities_sha256 independent_evidence_sha256')
    digest(approved)
    for key in ('source_sha256', 'previous_window_sha256', 'identities_sha256', 'independent_evidence_sha256'):
        digest(evidence[key])
    _require(type(evidence['version']) is int and evidence['version'] == 1
             and sha(_encode(evidence)) == approved
             and evidence['authority'] == str(budget.authority_root)
             and evidence['scope'] == budget.scope and evidence['billing_scope'] == budget.billing_scope
             and evidence['source_sha256'] == sha(source)
             and evidence['previous_window_sha256'] == sha(raw_window)
             and evidence['identities_sha256'] == sha(_encode(ids)))


def _candidate(budget, old, state, contract, price, previous):
    validate_recovery_contract(contract)
    _require(contract['corpus_sha256'] == sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes())
             and all(contract[k] == v for k, v in verified_snapshot_binding(budget).items()))
    used = {f['id'] for f in old['workflows']} | {a['id'] for a in old['legacy']['attempts']}
    used.add(previous['request']['invocation_id'])
    _require(contract['invocation_id'] not in used and isinstance(price, dict)
             and price.get('id') not in {p['id'] for p in old['pricing_revisions']})
    candidate = deepcopy(old)
    candidate.update(max_attempts=22, active_pricing=price['id'],
                     diagnostic_request_sha256=sha(_encode(contract)))
    candidate['pricing_revisions'].append(deepcopy(price))
    validator = copy(budget); validator.policy = BudgetPolicy(22)
    _, prices, after = validator._validate_document(candidate)
    _require(after == state)
    slots = _SLOTS[GRAPH]
    _require((len(slots), sum(s[2] for s in slots), sum(s[3] for s in slots)) == (7,184064,20736))
    cost = sum((_estimate(prices[price['id']], s[2], s[3]) for s in slots), Decimal(0))
    _require(state.reserved_usd + cost <= budget.policy.max_reserved_usd)
    return candidate, cost, validator


def _plan(budget, source, raw_window, candidate, contract, continuity, ids):
    return dict(version=3, kind=RECOVERY_KIND, authority=str(budget.authority_root), **ids,
        source_sha256=sha(source), candidate_sha256=sha(_encode(candidate)),
        scope=budget.scope, billing_scope=budget.billing_scope, model=MODEL_ID,
        old_cap=15, new_cap=22, cost_cap='0.100000',
        pricing_sha256=sha(_encode(candidate['pricing_revisions'][-1])), request=contract,
        previous_window_sha256=sha(raw_window), continuity_sha256=sha(_encode(continuity)))


def prepare_recovery(budget, contract, continuity, *, approved_continuity_sha256,
                     approved_max_attempts, approved_max_reserved_usd, pricing_revision,
                     verified_pricing_sha256, writers_stopped, recover_identity, dry_run=True):
    """Default read-only; independent approval evidence cannot be inferred from inodes."""
    with _operation(budget, writers_stopped, recover_identity):
        _require(type(approved_max_attempts) is int and approved_max_attempts == 22
                 and approved_max_reserved_usd == '0.100000' and type(dry_run) is bool)
        source, raw_window, previous, old, state, ids = _source(budget)
        _continuity(budget, continuity, approved_continuity_sha256, source, raw_window, ids)
        digest(verified_pricing_sha256)
        _require(sha(_encode(pricing_revision)) == verified_pricing_sha256)
        candidate, cost, _ = _candidate(budget, old, state, contract, pricing_revision, previous)
        plan = _plan(budget, source, raw_window, candidate, contract, continuity, ids)
        validate_plan(budget, plan)
        raw = _encode(plan); plan_id = sha(raw)
        directory = None
        if not dry_run:
            directory = budget.directory / ('pinecone-graph-recovery-' + plan_id)
            directory.mkdir(mode=0o700); _sync(budget.directory)
            for name, data in (('backup.v2.json', source), ('backup.window.json', raw_window),
                    ('candidate.v2.json', _encode(candidate)), ('continuity.json', _encode(continuity)),
                    ('plan.json', raw)):
                _write_private(budget, directory/name, data)
        _require(_identities(budget) == ids and budget._read_bytes() == source
                 and _read_private(budget, budget.directory/WINDOW_NAME) == raw_window)
        return AmendmentPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                state.attempts_consumed, str(state.reserved_usd), str(cost), directory)


def rebind_approval(plan_sha256):
    digest(plan_sha256)
    return sha(('pinecone-graph-rebind:' + plan_sha256).encode('ascii'))


def apply_recovery(budget, plan_id, *, approved_plan_sha256, approved_rebind_sha256,
                   writers_stopped, recover_identity):
    """Separate candidate/rebind approval. All checks and writes share the old lock."""
    with _operation(budget, writers_stopped, recover_identity):
        digest(plan_id)
        _require(approved_plan_sha256 == plan_id and approved_rebind_sha256 == rebind_approval(plan_id))
        directory = budget.directory / ('pinecone-graph-recovery-' + plan_id)
        _private_directory(budget, directory)
        raw = _read_private(budget, directory/'plan.json')
        _require(len(raw) <= 16384 and sha(raw) == plan_id)
        plan = _decode(raw); validate_plan(budget, plan)
        _require(plan['kind'] == RECOVERY_KIND and not os.path.lexists(directory/'apply.started'))
        source, raw_window, previous, old, state, ids = _source(budget)
        _require(_read_private(budget, directory/'backup.v2.json') == source
                 and _read_private(budget, directory/'backup.window.json') == raw_window)
        evidence = _decode(_read_private(budget, directory/'continuity.json'))
        _continuity(budget, evidence, plan['continuity_sha256'], source, raw_window, ids)
        candidate_raw = _read_private(budget, directory/'candidate.v2.json')
        candidate = _decode(candidate_raw)
        expected, _, validator = _candidate(budget, old, state, plan['request'],
                                            candidate['pricing_revisions'][-1], previous)
        _require(candidate_raw == _encode(expected)
                 and plan == _plan(budget, source, raw_window, expected, plan['request'], evidence, ids))

        def before_replace():
            _require(_identities(budget) == ids and budget._read_bytes() == source
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw_window)
            _write_private(budget, directory/'apply.started', _encode({'plan_sha256':plan_id}))
            next_window = directory/'window.next.json'
            _write_private(budget, next_window, raw)
            os.replace(next_window, budget.directory/WINDOW_NAME)
            _sync(directory); _sync(budget.directory)

        def verify():
            validate_plan(budget, plan)
            _require(budget._read_bytes() == candidate_raw and validator._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw)
            _write_private(budget, directory/'apply.completed', _encode({'plan_sha256':plan_id}))

        budget._settle(candidate_raw, before_replace=before_replace, verify=verify)
        return plan_id


def run_approval(plan_sha256):
    digest(plan_sha256)
    return sha(('pinecone-graph-run:' + plan_sha256).encode('ascii'))


def run_recovery(budget, request, *, plan_sha256, approved_run_sha256, question_sha256,
                 allow_live, other_writers_stopped):
    """One exact workflow, existing live gates and permits, no repair or retry."""
    from .live_graph import open_live_graph_service
    from .live_preflight import require_private_preview
    from .qa import QASuccess, QAFailure
    try:
        _require(allow_live is True and other_writers_stopped is True
                 and isinstance(budget, PersistentBudget) and budget.policy == BudgetPolicy(22)
                 and budget.before_request is require_private_preview
                 and isinstance(request, WorkflowRequest) and request.profile == GRAPH
                 and request.backend == 'pinecone' and request.mode == 'approved_only')
        digest(question_sha256)
        _require(approved_run_sha256 == run_approval(plan_sha256)
                 and sha(request.question.encode('utf-8')) == question_sha256)
        with budget.reserve(request, diagnostic_approval=plan_sha256) as reservation:
            reservation.check_setup(budget, profile=GRAPH, backend='pinecone')
            _require(reservation.envelope == (7,184064,20736))
            with open_live_graph_service(budget.root, backend='pinecone', profile=GRAPH,
                    budget=budget, reservation=reservation) as service:
                result = service.ask(request.question, mode=request.mode,
                    reference_date=request.reference_date, reservation=reservation)
            _require(isinstance(result, (QASuccess, QAFailure)))
        if isinstance(result, QASuccess):
            return DiagnosticOutcome(True, result.public.kind, 'accepted', result.public)
        return DiagnosticOutcome(False, 'error', 'workflow_rejected')
    except Exception:
        return DiagnosticOutcome(False, 'error', 'recovery_denied_or_failed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect')
    prepare = sub.add_parser('prepare')
    for name in ('contract', 'continuity', 'pricing-revision'):
        prepare.add_argument('--'+name, type=Path, required=True)
    prepare.add_argument('--approve-continuity', required=True)
    prepare.add_argument('--approve-max-attempts', type=int, required=True)
    prepare.add_argument('--approve-max-reserved-usd', required=True)
    prepare.add_argument('--verified-pricing-sha256', required=True)
    prepare.add_argument('--write-artifacts', action='store_true')
    apply = sub.add_parser('apply')
    apply.add_argument('--plan', required=True)
    apply.add_argument('--approve-plan', required=True)
    apply.add_argument('--approve-rebind', required=True)
    for command in (inspect, prepare, apply):
        command.add_argument('--writers-stopped', action='store_true', required=True)
        command.add_argument('--recover-identity', action='store_true', required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--approve-run', required=True)
    run.add_argument('--invocation-id', required=True)
    run.add_argument('--question-file', type=Path, required=True)
    run.add_argument('--question-sha256', required=True)
    run.add_argument('--mode', choices=('approved_only',), required=True)
    run.add_argument('--backend', choices=('pinecone',), required=True)
    run.add_argument('--profile', choices=(GRAPH,), required=True)
    run.add_argument('--reference-date', required=True)
    run.add_argument('--allow-live', action='store_true', required=True)
    run.add_argument('--other-writers-stopped', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(args.root)

        def private_input(path):
            _require(path.is_relative_to(budget.root / '.local'))
            return _read_private(budget, path, root=budget.root)

        if args.command == 'inspect':
            print(_encode(inspect_recovery(budget, writers_stopped=args.writers_stopped,
                                          recover_identity=args.recover_identity)).decode())
        elif args.command == 'prepare':
            contract, continuity, price = (_decode(private_input(p)) for p in
                                          (args.contract, args.continuity, args.pricing_revision))
            preview = prepare_recovery(budget, contract, continuity,
                approved_continuity_sha256=args.approve_continuity,
                approved_max_attempts=args.approve_max_attempts,
                approved_max_reserved_usd=args.approve_max_reserved_usd,
                pricing_revision=price, verified_pricing_sha256=args.verified_pricing_sha256,
                writers_stopped=args.writers_stopped, recover_identity=args.recover_identity,
                dry_run=not args.write_artifacts)
            print(_encode({k:v for k,v in asdict(preview).items() if k != 'artifact_directory'}).decode())
        elif args.command == 'apply':
            apply_recovery(budget, args.plan, approved_plan_sha256=args.approve_plan,
                approved_rebind_sha256=args.approve_rebind, writers_stopped=args.writers_stopped,
                recover_identity=args.recover_identity)
            print('Recovery applied. Ordinary entrypoints remain blocked; live execution needs separate approval.')
        else:
            raw = private_input(args.question_file)
            digest(args.question_sha256)
            _require(len(raw) <= 64000 and sha(raw) == args.question_sha256)
            request = WorkflowRequest(args.invocation_id, args.profile, raw.decode('utf-8'),
                                      args.mode, args.backend, date.fromisoformat(args.reference_date))
            outcome = run_recovery(budget, request, plan_sha256=args.plan,
                approved_run_sha256=args.approve_run, question_sha256=args.question_sha256,
                allow_live=args.allow_live, other_writers_stopped=args.other_writers_stopped)
            print(f'Recovery workflow: {outcome.kind}; {outcome.code}. No further workflow attempted.')
            return 0 if outcome.accepted else 1
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        print('Recovery denied or uncertain. Keep writers stopped; review private evidence. No automatic retry.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
