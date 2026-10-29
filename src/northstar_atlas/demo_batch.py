"""Private five-submission Standard RAG batches; no initialization or refunds.

Only 2->7 and separately approved 7->12 are supported. Pricing is never edited.
Artifacts and authority remain private. Importing this module performs no I/O.
"""
import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
from dataclasses import asdict
from decimal import Decimal
import os
from pathlib import Path
import re

from .budget_amendment import AmendmentPreview
from .budget_transition import _encode, _decode, _read_private, _write_private, _private_directory, _sync
from .diagnostic_window import (WINDOW_NAME, DEMO_KIND, DEMO_BILLING_SCOPE,
                                digest, identity, sha, validate_plan)
from .live_budget import (BudgetError, BudgetPolicy, PersistentBudget, LINEAR,
                          _fields, _estimate, _require, _SLOTS, configured_budget)
from .llm import MODEL_ID
from .project import contained_path

KIND = 'v2-five-question-demo-batch'


def batch_contract(budget, batch_id):
    return dict(batch_id=batch_id, profile=LINEAR, backend='pinecone', model=MODEL_ID,
        modes=['approved_only', 'draft_review'], submissions=5,
        corpus_sha256=sha(contained_path(budget.root, 'data/corpus_manifest.json').read_bytes()))


def _contract(value):
    _fields(value, 'batch_id profile backend model modes submissions corpus_sha256')
    _require(isinstance(value['batch_id'], str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', value['batch_id'])
             and value['profile'] == LINEAR and value['backend'] == 'pinecone' and value['model'] == MODEL_ID
             and value['modes'] == ['approved_only', 'draft_review']
             and type(value['submissions']) is int and value['submissions'] == 5)
    digest(value['corpus_sha256'])


def validate_batch_plan(budget, plan):
    _fields(plan, 'version kind authority authority_identity local_identity directory_identity lock_identity '
        'ledger_identity source_sha256 candidate_sha256 previous_window_sha256 scope billing_scope model '
        'old_cap new_cap cost_cap max_call_reserved_usd pricing_sha256 request')
    _require(type(plan['version']) is int and plan['version'] == 5 and plan['kind'] == KIND
        and plan['authority'] == str(budget.authority_root) and plan['scope'] == budget.scope
        and plan['billing_scope'] == budget.billing_scope == DEMO_BILLING_SCOPE and plan['model'] == MODEL_ID
        and type(plan['old_cap']) is int and type(plan['new_cap']) is int
        and (plan['old_cap'], plan['new_cap']) in ((2, 7), (7, 12))
        and plan['cost_cap'] == '0.100000' and plan['max_call_reserved_usd'] == '0.010000')
    for key in ('source_sha256', 'candidate_sha256', 'previous_window_sha256', 'pricing_sha256'): digest(plan[key])
    for key in ('authority_identity', 'local_identity', 'directory_identity', 'lock_identity', 'ledger_identity'):
        pair = plan[key]
        _require(isinstance(pair, list) and len(pair) == 2 and all(type(n) is int and n >= 0 for n in pair))
    _contract(plan['request'])


def _active_price(ledger):
    return next(p for p in ledger['pricing_revisions'] if p['id'] == ledger['active_pricing'])


def _cost(price):
    cost = sum((_estimate(price, s[2], s[3]) for s in _SLOTS[LINEAR]), Decimal(0))
    _require(cost <= Decimal('0.010000'))
    return cost


def _progress(budget, plan, ledger):
    """Under the ledger lock: approved immutable prefix plus bounded new flows."""
    validate_plan(budget, plan)
    _require(plan['kind'] == KIND and budget.policy == BudgetPolicy(plan['new_cap'])
             and plan['request'] == batch_contract(budget, plan['request']['batch_id']))
    directory = budget.directory / ('demo-batch-' + sha(_encode(plan)))
    _private_directory(budget, directory)
    raw = _read_private(budget, directory/'candidate.v2.json')
    _require(sha(raw) == plan['candidate_sha256'])
    base = _decode(raw)
    budget._validate_document(base)  # Current price and authoritative schema remain mandatory.
    _, prices, state = budget._validate_document(ledger)
    count = len(base['workflows'])
    prefix = deepcopy(ledger); prefix['workflows'] = prefix['workflows'][:count]
    _require(prefix == base and base['diagnostic_request_sha256'] == sha(_encode(plan['request']))
             and sha(_encode(_active_price(ledger))) == plan['pricing_sha256'])
    tail = ledger['workflows'][count:]
    _require(len(tail) <= 5)
    for flow in tail:
        _require(flow['profile'] == LINEAR and flow['backend'] == 'pinecone'
                 and flow['mode'] in plan['request']['modes']
                 and flow['price_id'] == ledger['active_pricing'])
    cost = _cost(prices[ledger['active_pricing']])
    return 5-len(tail), state, cost


def check_batch_request(budget, plan, request, corpus, ledger):
    remaining_count, _, _ = _progress(budget, plan, ledger)
    _require(remaining_count > 0 and request.profile == LINEAR and request.backend == 'pinecone'
             and request.mode in plan['request']['modes'] and corpus == plan['request']['corpus_sha256'])


def remaining(budget, plan_sha256):
    """Durable submissions remaining, including conservative zero-call failures."""
    with budget._locked(diagnostic_approval=plan_sha256) as plan:
        ledger, _, _ = budget._read()
        count, state, cost = _progress(budget, plan, ledger)
        if state.attempts_consumed >= budget.policy.max_attempts or state.reserved_usd + cost > budget.policy.max_reserved_usd:
            return 0
        return count


@contextmanager
def _operation(budget, previous, writers_stopped):
    try:
        _require(writers_stopped is True and budget.policy in (BudgetPolicy(2), BudgetPolicy(7))
                 and budget.billing_scope == DEMO_BILLING_SCOPE)
        with budget._locked(diagnostic_approval=previous) as plan:
            _require(plan['kind'] == (DEMO_KIND if budget.policy.max_attempts == 2 else KIND))
            yield plan
    except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
        raise BudgetError('batch amendment denied or uncertain; keep writers stopped') from None


def _candidate(budget, source, previous, contract):
    _contract(contract)
    _require(contract == batch_contract(budget, contract['batch_id']))
    old, _, state = budget._validate_document(_decode(source))
    cap = budget.policy.max_attempts
    _require(state.version == 2 and old['max_reserved_usd'] == '0.100000'
        and (old['legacy'] is None or old['legacy']['attempts'] == [])
        and old.get('diagnostic_request_sha256') == sha(_encode(previous['request']))
        and all(f['profile'] == LINEAR and f['status'] in ('closed', 'uncertain') for f in old['workflows']))
    if cap == 2:
        _require(state.attempts_consumed == 2 and len(old['workflows']) == 2
                 and all(len(f['calls']) == 1 for f in old['workflows'])
                 and old['workflows'][-1]['id'] == previous['request']['invocation_id'])
    else:
        _require(_progress(budget, previous, old)[0] == 0
                 and contract['batch_id'] != previous['request']['batch_id'])
    candidate = deepcopy(old)
    candidate.update(max_attempts=cap+5, diagnostic_request_sha256=sha(_encode(contract)))
    validator = copy(budget); validator.policy = BudgetPolicy(cap+5)
    _, _, after = validator._validate_document(candidate)
    _require(after == state)
    cost = _cost(_active_price(candidate))
    _require(state.reserved_usd + 5*cost <= budget.policy.max_reserved_usd)
    return candidate, state, cost, validator


def _plan(budget, source, candidate, contract, previous):
    return dict(version=5, kind=KIND, authority=str(budget.authority_root),
        authority_identity=identity(budget.authority_root), local_identity=identity(budget.authority_root/'.local'),
        directory_identity=identity(budget.directory), lock_identity=identity(budget.lock_path),
        ledger_identity=identity(budget.path), source_sha256=sha(source), candidate_sha256=sha(_encode(candidate)),
        previous_window_sha256=previous, scope=budget.scope, billing_scope=budget.billing_scope, model=MODEL_ID,
        old_cap=budget.policy.max_attempts, new_cap=budget.policy.max_attempts+5, cost_cap='0.100000',
        max_call_reserved_usd='0.010000', pricing_sha256=sha(_encode(_active_price(candidate))), request=contract)


def prepare_batch(budget, contract, *, previous_plan_sha256, confirmed_billing_scope,
                  approved_max_attempts, approved_max_reserved_usd, writers_stopped,
                  verified_pricing_sha256, dry_run=True):
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        _require(confirmed_billing_scope == budget.billing_scope and type(approved_max_attempts) is int
                 and approved_max_attempts == budget.policy.max_attempts+5
                 and approved_max_reserved_usd == '0.100000' and type(dry_run) is bool)
        digest(verified_pricing_sha256)
        source = budget._read_bytes(); window = _read_private(budget, budget.directory/WINDOW_NAME)
        _require(sha(window) == previous_plan_sha256)
        candidate, state, cost, _ = _candidate(budget, source, previous, contract)
        _require(sha(_encode(_active_price(candidate))) == verified_pricing_sha256)
        plan = _plan(budget, source, candidate, contract, previous_plan_sha256)
        raw = _encode(plan); plan_id = sha(raw); directory = None
        if not dry_run:
            directory = budget.directory/('demo-batch-' + plan_id)
            directory.mkdir(mode=0o700); _sync(budget.directory)
            for name, data in (('backup.v2.json', source), ('backup.window.json', window),
                               ('candidate.v2.json', _encode(candidate)), ('plan.json', raw)):
                _write_private(budget, directory/name, data)
        return AmendmentPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                state.attempts_consumed, str(state.reserved_usd), str(cost*5), directory)


def apply_batch(budget, plan_id, *, approved_plan_sha256, previous_plan_sha256, writers_stopped):
    with _operation(budget, previous_plan_sha256, writers_stopped) as previous:
        digest(plan_id); digest(approved_plan_sha256); _require(plan_id == approved_plan_sha256)
        directory = budget.directory/('demo-batch-' + plan_id); _private_directory(budget, directory)
        raw = _read_private(budget, directory/'plan.json')
        _require(len(raw) <= 16384 and sha(raw) == plan_id)
        plan = _decode(raw); validate_plan(budget, plan)
        _require(plan['kind'] == KIND and plan['previous_window_sha256'] == previous_plan_sha256
                 and not os.path.lexists(directory/'apply.started'))
        source = _read_private(budget, directory/'backup.v2.json')
        window = _read_private(budget, directory/'backup.window.json')
        candidate_raw = _read_private(budget, directory/'candidate.v2.json')
        _require(sha(source) == plan['source_sha256'] and sha(candidate_raw) == plan['candidate_sha256']
                 and sha(window) == previous_plan_sha256)
        expected, state, _, validator = _candidate(budget, source, previous, plan['request'])
        _require(candidate_raw == _encode(expected)
                 and plan == _plan(budget, source, expected, plan['request'], previous_plan_sha256))

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
            _require(budget._read_bytes() == candidate_raw and validator._read()[2] == state
                     and _read_private(budget, budget.directory/WINDOW_NAME) == raw)
            _write_private(budget, directory/'apply.completed', _encode({'plan_sha256': plan_id}))
        budget._settle(candidate_raw, before_replace=before_replace, verify=verify)
        return plan_id


def run_approval(plan_sha256):
    digest(plan_sha256)
    return sha(('interactive-demo-batch-run:' + plan_sha256).encode('ascii'))


def activation():
    """Trusted process configuration, never chat input or a browser session value."""
    _require(all(os.environ.get(name) == '1' for name in
        ('NORTHSTAR_DEMO_BATCH_ENABLED', 'NORTHSTAR_CONTROLLED_LIVE', 'NORTHSTAR_PRIVATE_PREVIEW_CONFIRMED'))
        and os.environ.get('NORTHSTAR_UI_ENABLE_GRAPH', '0') == '0')
    plan = os.environ.get('NORTHSTAR_DEMO_BATCH_PLAN', '')
    _require(os.environ.get('NORTHSTAR_DEMO_BATCH_RUN_APPROVAL') == run_approval(plan))
    return plan


def run_question(budget, request, *, opener, limits):
    """One reservation-bound QA call, captured after settlement before presentation."""
    from .live_preflight import require_private_preview
    from .qa import QASuccess, QAFailure
    from .result_capture import capture_result
    plan = activation()
    _require(isinstance(budget, PersistentBudget) and budget.before_request is require_private_preview
             and request.profile == LINEAR and request.backend == 'pinecone')

    def execute_once():
        with budget.reserve(request, diagnostic_approval=plan) as reservation:
            reservation.check_setup(budget, profile=LINEAR, backend='pinecone')
            _require(reservation.envelope == (1, 30720, 2048))
            require_private_preview()
            with opener(budget.root, backend='pinecone', allow_paid=True, model=MODEL_ID,
                        budget=budget, reservation=reservation) as service:
                reservation.check_setup(budget, profile=LINEAR, backend='pinecone')
                result = service.ask(request.question, mode=request.mode, top_k=5,
                    reference_date=request.reference_date, limits=limits, reservation=reservation)
            _require(isinstance(result, (QASuccess, QAFailure)))
        return result
    return capture_result(execute_once, root=budget.root, invocation_id=request.invocation_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--batch-id', required=True)
    prepare.add_argument('--confirmed-billing-scope', required=True)
    prepare.add_argument('--approve-max-attempts', type=int, required=True)
    prepare.add_argument('--approve-max-reserved-usd', required=True)
    prepare.add_argument('--verified-pricing-sha256', required=True)
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
            preview = prepare_batch(budget, batch_contract(budget, args.batch_id),
                previous_plan_sha256=args.previous_plan, confirmed_billing_scope=args.confirmed_billing_scope,
                approved_max_attempts=args.approve_max_attempts, approved_max_reserved_usd=args.approve_max_reserved_usd,
                verified_pricing_sha256=args.verified_pricing_sha256, writers_stopped=args.writers_stopped,
                dry_run=not args.write_artifacts)
            print(_encode({k: v for k, v in asdict(preview).items() if k != 'artifact_directory'}).decode())
        else:
            apply_batch(budget, args.plan, approved_plan_sha256=args.approve,
                        previous_plan_sha256=args.previous_plan, writers_stopped=args.writers_stopped)
            print('Batch amendment applied; separate live batch authorization remains required.')
        return 0
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        print('Batch denied or uncertain. Keep writers stopped and review private artifacts.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
