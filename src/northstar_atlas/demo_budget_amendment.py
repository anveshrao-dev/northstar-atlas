"""Operator-only dedicated demo amendment: one lifetime slot to two.

No initialization, history reconciliation, refunds or rollback. The historical
graph authority is not accepted; this tool operates only on the existing demo.
"""
import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict
from decimal import Decimal
import os
from pathlib import Path

from .budget_amendment import AmendmentPreview

# Reuse create-only private artifact primitives, never the v1 transition API.
from .budget_transition import _encode, _decode, _read_private, _write_private, _private_directory, _sync
from .diagnostic_window import (WINDOW_NAME, DEMO_KIND, DEMO_BILLING_SCOPE, digest, identity, sha,
                                validate_contract, validate_plan)
from .live_budget import (BudgetError, BudgetPolicy, PersistentBudget, LINEAR,
                          _estimate, _require, _SLOTS, configured_budget)
from .llm import MODEL_ID
from .project import contained_path


@contextmanager
def _operation(budget, writers_stopped):
    try:
        _require(writers_stopped is True and budget.policy == BudgetPolicy(1)
                 and budget.billing_scope == DEMO_BILLING_SCOPE)
        with budget._locked() as window:
            _require(window is None)
            yield
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('amendment denied or uncertain; keep writers stopped and review private artifacts') from None


def _candidate(budget, source, pricing_revision, contract):
    validate_contract(contract, profile=LINEAR)
    _require(contract['mode'] == 'approved_only')
    old, _, state = budget._validate_document(_decode(source), current=False)
    _require(state.version == 2 and budget.policy == BudgetPolicy(1)
             and budget.billing_scope == DEMO_BILLING_SCOPE
             and old['max_attempts'] == 1 and old['max_reserved_usd'] == '0.100000'
             and state.attempts_consumed == 1 and state.reserved_usd == Decimal('0.003687')
             and old['legacy'] is None and len(old['workflows']) == 1
             and 'diagnostic_request_sha256' not in old)
    first = old['workflows'][0]
    _require(first['profile'] == LINEAR and first['status'] in ('closed', 'uncertain')
             and len(first['calls']) == 1 and first['id'] != contract['invocation_id'])
    candidate = deepcopy(old)
    candidate['max_attempts'] = 2
    candidate['diagnostic_request_sha256'] = sha(_encode(contract))
    if pricing_revision is not None:
        _require(isinstance(pricing_revision, dict)
                 and pricing_revision.get('id') not in {p['id'] for p in old['pricing_revisions']})
        candidate['pricing_revisions'].append(deepcopy(pricing_revision))
        candidate['active_pricing'] = pricing_revision['id']
    validator = copy(budget)
    validator.policy = BudgetPolicy(2)
    _, prices, after = validator._validate_document(candidate)
    _require(after == state)
    price = prices[candidate['active_pricing']]
    cost = sum((_estimate(price, s[2], s[3]) for s in _SLOTS[LINEAR]), Decimal(0))
    _require(cost <= Decimal('0.010000')
             and state.reserved_usd + cost <= budget.policy.max_reserved_usd)
    return candidate, state, cost, validator


def _plan(budget, source, candidate, contract):
    active = next(p for p in candidate['pricing_revisions'] if p['id'] == candidate['active_pricing'])
    return dict(version=4, kind=DEMO_KIND, authority=str(budget.authority_root),
        authority_identity=identity(budget.authority_root),
        local_identity=identity(budget.authority_root/'.local'),
        directory_identity=identity(budget.directory), lock_identity=identity(budget.lock_path),
        ledger_identity=identity(budget.path), source_sha256=sha(source), candidate_sha256=sha(_encode(candidate)),
        scope=budget.scope, billing_scope=budget.billing_scope, model=MODEL_ID,
        old_cap=1, new_cap=2, cost_cap='0.100000', max_call_reserved_usd='0.010000', pricing_sha256=sha(_encode(active)), request=contract)


def prepare_amendment(budget, contract, *, confirmed_billing_scope, approved_max_attempts,
                      approved_max_reserved_usd, writers_stopped, pricing_revision=None,
                      verified_pricing_sha256=None, dry_run=True):
    """Default is read-only. Artifact preparation never replaces the ledger."""
    with _operation(budget, writers_stopped):
        _require(confirmed_billing_scope == budget.billing_scope and type(approved_max_attempts) is int
                 and approved_max_attempts == 2 and approved_max_reserved_usd == '0.100000'
                 and type(dry_run) is bool)
        validate_contract(contract, profile=LINEAR)
        _require(contract['mode'] == 'approved_only')
        _require(contract['corpus_sha256'] == sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))
        digest(verified_pricing_sha256)
        source = budget._read_bytes()
        candidate, state, cost, _ = _candidate(budget, source, pricing_revision, contract)
        active = next(p for p in candidate['pricing_revisions'] if p['id'] == candidate['active_pricing'])
        _require(sha(_encode(active)) == verified_pricing_sha256)
        _require(all(f['id'] != contract['invocation_id'] for f in candidate['workflows']))
        plan = _plan(budget, source, candidate, contract)
        raw = _encode(plan); plan_id = sha(raw)
        directory = None
        if not dry_run:
            directory = budget.directory / ('demo-amendment-' + plan_id)
            directory.mkdir(mode=0o700)
            _sync(budget.directory)
            for name, data in (('backup.v2.json', source), ('candidate.v2.json', _encode(candidate)), ('plan.json', raw)):
                _write_private(budget, directory/name, data)
        return AmendmentPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                state.attempts_consumed, str(state.reserved_usd), str(cost), directory)


def apply_amendment(budget, plan_id, *, approved_plan_sha256, writers_stopped):
    """Separate exact-candidate approval; holds the same lock through readback."""
    with _operation(budget, writers_stopped):
        digest(plan_id); digest(approved_plan_sha256)
        _require(plan_id == approved_plan_sha256)
        directory = budget.directory / ('demo-amendment-' + plan_id)
        _private_directory(budget, directory)
        raw_plan = _read_private(budget, directory/'plan.json')
        _require(sha(raw_plan) == plan_id and len(raw_plan) <= 16384)
        plan = _decode(raw_plan); validate_plan(budget, plan)
        _require(plan['kind'] == DEMO_KIND)
        _require(not os.path.lexists(directory/'apply.started'))
        source = _read_private(budget, directory/'backup.v2.json')
        candidate_raw = _read_private(budget, directory/'candidate.v2.json')
        _require(sha(source) == plan['source_sha256'] and sha(candidate_raw) == plan['candidate_sha256']
                 and budget._read_bytes() == source and identity(budget.path) == plan['ledger_identity'])
        candidate = _decode(candidate_raw); old = _decode(source)
        revisions = candidate['pricing_revisions']
        _require(revisions[:len(old['pricing_revisions'])] == old['pricing_revisions']
                 and len(revisions) in (len(old['pricing_revisions']), len(old['pricing_revisions'])+1))
        added = revisions[-1] if len(revisions) > len(old['pricing_revisions']) else None
        expected, state, _, validator = _candidate(budget, source, added, plan['request'])
        _require(candidate == expected and candidate_raw == _encode(expected)
                 and plan == _plan(budget, source, candidate, plan['request']))
        _require(plan['request']['corpus_sha256'] == sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))

        def before_replace():
            validate_plan(budget, plan)
            _require(budget._read_bytes() == source and identity(budget.path) == plan['ledger_identity'])
            _write_private(budget, directory/'apply.started', _encode({'plan_sha256': plan_id}))
            # Keep ordinary entrypoints blocked even after settlement and restart.
            _write_private(budget, budget.directory/WINDOW_NAME, raw_plan)

        def verify():
            validate_plan(budget, plan)
            _require(budget._read_bytes() == candidate_raw and validator._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw_plan)
            _write_private(budget, directory/'apply.completed', _encode({'plan_sha256': plan_id}))

        budget._settle(candidate_raw, before_replace=before_replace, verify=verify)
        return plan_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--contract', type=Path, required=True, help='Private JSON request contract; no raw question')
    prepare.add_argument('--confirmed-billing-scope', required=True)
    prepare.add_argument('--approve-max-attempts', type=int, required=True)
    prepare.add_argument('--approve-max-reserved-usd', required=True)
    prepare.add_argument('--pricing-revision', type=Path)
    prepare.add_argument('--verified-pricing-sha256', required=True)
    prepare.add_argument('--write-artifacts', action='store_true')
    apply = sub.add_parser('apply')
    apply.add_argument('--plan', required=True)
    apply.add_argument('--approve', required=True)
    for command in (prepare, apply): command.add_argument('--writers-stopped', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(args.root)
        if args.command == 'prepare':
            for path in (args.contract, args.pricing_revision):
                if path is not None: _require(path.is_relative_to(budget.root/'.local'))
            contract = _decode(_read_private(budget, args.contract, root=budget.root))
            price = None if args.pricing_revision is None else _decode(_read_private(budget, args.pricing_revision, root=budget.root))
            preview = prepare_amendment(budget, contract, confirmed_billing_scope=args.confirmed_billing_scope,
                approved_max_attempts=args.approve_max_attempts, approved_max_reserved_usd=args.approve_max_reserved_usd,
                writers_stopped=args.writers_stopped, pricing_revision=price,
                verified_pricing_sha256=args.verified_pricing_sha256, dry_run=not args.write_artifacts)
            # Print digests/counts only; private identity and contents stay in artifacts.
            print(_encode({k: v for k, v in asdict(preview).items() if k != 'artifact_directory'}).decode())
        else:
            apply_amendment(budget, args.plan, approved_plan_sha256=args.approve, writers_stopped=args.writers_stopped)
            print('Demo amendment applied; separate one-shot run authorization remains required.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print('Amendment denied or uncertain. Keep writers stopped; review private artifacts.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
