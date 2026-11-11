"""Operator-only pricing successor for an unused, approved 7->12 demo batch.

No initialization, cap extension, refunds, provider lookup or automatic recovery.
Existing artifacts remain immutable; a distinct approval replaces the live window.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path

from . import demo_batch as batch
from .budget_amendment import AmendmentPreview
from .budget_transition import _encode, _decode, _read_private, _write_private, _private_directory, _sync
from .diagnostic_window import WINDOW_NAME, DEMO_BILLING_SCOPE, digest, identity, sha, validate_plan
from .live_budget import BudgetError, BudgetPolicy, LINEAR, _require, configured_budget

KIND = 'v2-unused-demo-batch-pricing-renewal'


def validate_renewal_plan(budget, plan):
    """Same identity/approval fields as a batch, with a distinct version and kind."""
    _require(type(plan.get('version')) is int and plan['version'] == 6
             and plan.get('kind') == KIND and type(plan.get('old_cap')) is int
             and plan['old_cap'] == 12 and type(plan.get('new_cap')) is int and plan['new_cap'] == 12)
    original_shape = dict(plan, version=5, kind=batch.KIND, old_cap=7)
    batch.validate_batch_plan(budget, original_shape)


def _directory(budget, plan_id):
    digest(plan_id)
    return budget.directory/('demo-batch-pricing-' + plan_id)


@contextmanager
def _operation(budget, previous, writers_stopped):
    try:
        _require(writers_stopped is True and budget.policy == BudgetPolicy(12)
                 and budget.billing_scope == DEMO_BILLING_SCOPE)
        with budget._locked(diagnostic_approval=previous) as plan:
            _require(plan['kind'] == batch.KIND and plan['old_cap'] == 7 and plan['new_cap'] == 12)
            yield plan
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('pricing renewal denied or uncertain; keep writers stopped') from None


def _candidate(budget, source, previous, price):
    validate_plan(budget, previous)
    _require(previous['kind'] == batch.KIND and previous['old_cap'] == 7 and previous['new_cap'] == 12
             and budget.policy == BudgetPolicy(12)
             and previous['request'] == batch.batch_contract(budget, previous['request']['batch_id']))
    original_directory = budget.directory/('demo-batch-' + sha(_encode(previous)))
    _private_directory(budget, original_directory)
    original = _read_private(budget, original_directory/'candidate.v2.json')
    _require(sha(original) == previous['candidate_sha256'] and source == original)
    # Expiry is allowed ONLY when inspecting the immutable historical donor.
    old, _, state = budget._validate_document(_decode(source), current=False)
    _require(len(old['workflows']) == 7 and old['max_reserved_usd'] == '0.100000'
             and (old['legacy'] is None or old['legacy']['attempts'] == [])
             and old['diagnostic_request_sha256'] == sha(_encode(previous['request']))
             and sha(_encode(batch._active_price(old))) == previous['pricing_sha256']
             and all(f['profile'] == LINEAR and f['backend'] == 'pinecone'
                     and f['status'] in ('closed', 'uncertain') for f in old['workflows']))
    candidate = deepcopy(old)
    candidate['pricing_revisions'].append(deepcopy(price))
    candidate['active_pricing'] = price['id']
    # The authoritative validator rejects reused IDs, invalid revisions and expiry.
    _, _, after = budget._validate_document(candidate)
    _require(after == state)
    cost = batch._cost(price)
    _require(state.reserved_usd + 5*cost <= budget.policy.max_reserved_usd)
    return candidate, state, cost


def _plan(budget, source, candidate, previous, previous_id):
    plan = batch._plan(budget, source, candidate, previous['request'], previous_id)
    plan.update(version=6, kind=KIND, old_cap=12, new_cap=12)
    return plan


def load_baseline(budget, plan):
    """Under the existing lock, verify the original approval and successor chain."""
    validate_plan(budget, plan)
    _require(plan['kind'] == KIND)
    directory = _directory(budget, sha(_encode(plan)))
    _private_directory(budget, directory)
    saved_plan = _read_private(budget, directory/'plan.json')
    source = _read_private(budget, directory/'backup.v2.json')
    window = _read_private(budget, directory/'backup.window.json')
    raw = _read_private(budget, directory/'candidate.v2.json')
    _require(saved_plan == _encode(plan) and sha(source) == plan['source_sha256']
             and sha(window) == plan['previous_window_sha256'] and len(window) <= 16384
             and sha(raw) == plan['candidate_sha256'])
    previous = _decode(window)
    candidate = _decode(raw)
    price = batch._active_price(candidate)
    expected, state, cost = _candidate(budget, source, previous, price)
    _require(raw == _encode(expected) and plan['request'] == previous['request']
             and plan['pricing_sha256'] == sha(_encode(price)))
    return candidate, state, cost


def prepare_renewal(budget, price, *, previous_plan_sha256, confirmed_billing_scope,
                    writers_stopped, verified_pricing_sha256, dry_run=True):
    """Prepare only. verified_pricing_sha256 attests independent operator review."""
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        digest(verified_pricing_sha256)
        _require(confirmed_billing_scope == budget.billing_scope and type(dry_run) is bool
                 and verified_pricing_sha256 == sha(_encode(price)))
        source = budget._read_bytes()
        window = _read_private(budget, budget.directory/WINDOW_NAME)
        _require(sha(window) == previous_plan_sha256)
        candidate, state, cost = _candidate(budget, source, previous, price)
        plan = _plan(budget, source, candidate, previous, previous_plan_sha256)
        raw = _encode(plan); plan_id = sha(raw); directory = None
        if not dry_run:
            directory = _directory(budget, plan_id)
            directory.mkdir(mode=0o700); _sync(budget.directory)
            for name, data in (('backup.v2.json', source), ('backup.window.json', window),
                               ('candidate.v2.json', _encode(candidate)), ('plan.json', raw)):
                _write_private(budget, directory/name, data)
        return AmendmentPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                state.attempts_consumed, str(state.reserved_usd), str(5*cost), directory)


def apply_renewal(budget, plan_id, *, approved_plan_sha256, previous_plan_sha256, writers_stopped):
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        digest(plan_id); digest(approved_plan_sha256); _require(plan_id == approved_plan_sha256)
        directory = _directory(budget, plan_id); _private_directory(budget, directory)
        raw = _read_private(budget, directory/'plan.json')
        _require(len(raw) <= 16384 and sha(raw) == plan_id)
        plan = _decode(raw)
        candidate, state, _ = load_baseline(budget, plan)
        source = _read_private(budget, directory/'backup.v2.json')
        window = _read_private(budget, directory/'backup.window.json')
        candidate_raw = _encode(candidate)
        _require(plan == _plan(budget, source, candidate, previous, previous_plan_sha256)
                 and not os.path.lexists(directory/'apply.started'))

        def unchanged():
            validate_plan(budget, plan)
            _require(budget._read_bytes() == source and identity(budget.path) == plan['ledger_identity']
                     and _read_private(budget, budget.directory/WINDOW_NAME) == window)
        unchanged()

        def before_replace():
            unchanged()
            _write_private(budget, directory/'apply.started', _encode({'plan_sha256': plan_id}))
            next_window = directory/'window.next.json'; _write_private(budget, next_window, raw)
            os.replace(next_window, budget.directory/WINDOW_NAME)
            _sync(directory); _sync(budget.directory)

        def verify():
            validate_plan(budget, plan)
            _require(budget._read_bytes() == candidate_raw and budget._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw)
            _write_private(budget, directory/'apply.completed', _encode({'plan_sha256': plan_id}))
        budget._settle(candidate_raw, before_replace=before_replace, verify=verify)
        return plan_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--pricing', type=Path, required=True)
    prepare.add_argument('--verified-pricing-sha256', required=True)
    prepare.add_argument('--confirmed-billing-scope', required=True)
    prepare.add_argument('--write-artifacts', action='store_true')
    apply = sub.add_parser('apply')
    apply.add_argument('--plan', required=True); apply.add_argument('--approve', required=True)
    for command in (prepare, apply):
        command.add_argument('--previous-plan', required=True)
        command.add_argument('--writers-stopped', action='store_true', required=True)
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(args.root)
        if args.command == 'prepare':
            pricing_raw = _read_private(budget, args.pricing)
            _require(len(pricing_raw) <= 16384)
            preview = prepare_renewal(budget, _decode(pricing_raw),
                previous_plan_sha256=args.previous_plan, writers_stopped=args.writers_stopped,
                confirmed_billing_scope=args.confirmed_billing_scope,
                verified_pricing_sha256=args.verified_pricing_sha256, dry_run=not args.write_artifacts)
            print(_encode({k:v for k,v in asdict(preview).items() if k != 'artifact_directory'}).decode())
        else:
            apply_renewal(budget, args.plan, approved_plan_sha256=args.approve,
                          previous_plan_sha256=args.previous_plan, writers_stopped=args.writers_stopped)
            print('Pricing successor applied; separate live batch authorization remains required.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        print('Pricing renewal denied or uncertain. Keep writers stopped and review private artifacts.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
