"""Operator-only v1 → v2 transition. Never imported by a live request path.

No ledger initialization, automatic recovery, pricing lookup or provider access.
All writers must be stopped independently; a lock cannot prove shutdown.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .live_budget import (
    BudgetError, PersistentBudget, _fields, _integer, _money, _require, _unique,
    configured_budget,
)
from .llm import MODEL_ID
from .project import contained_path


def _encode(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _decode(raw: bytes) -> dict:
    return json.loads(raw, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value: str) -> None:
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value))


@dataclass(frozen=True)
class TransitionPreview:
    plan_sha256: str
    candidate_sha256: str
    source_sha256: str
    attempts_preserved: int
    reserved_usd: str
    artifact_directory: Path | None


@contextmanager
def _operation(budget: PersistentBudget, writers_stopped: bool):
    try:
        _require(writers_stopped is True)
        with budget._locked():
            yield
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        raise BudgetError('transition denied or uncertain; keep writers stopped and review private artifacts') from None


def _private_directory(budget: PersistentBudget, path: Path) -> None:
    contained_path(budget.authority_root, str(path.relative_to(budget.authority_root)))
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
             and stat.S_IMODE(info.st_mode) == 0o700)


def _read_private(budget: PersistentBudget, path: Path, *, root: Path | None = None) -> bytes:
    base = budget.authority_root if root is None else root
    contained_path(base, str(path.relative_to(base)))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        budget._opened_file(stream.fileno(), path)
        raw = stream.read(4_000_001)
    _require(len(raw) <= 4_000_000)
    return raw


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_private(budget: PersistentBudget, path: Path, raw: bytes) -> None:
    """Create-only artifacts; failed/incomplete artifacts are never overwritten."""
    budget._paths()
    _private_directory(budget, path.parent)
    contained_path(budget.authority_root, str(path.relative_to(budget.authority_root)))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _require(_read_private(budget, path) == raw)
    _sync(path.parent)


def _legacy_source(budget: PersistentBudget, raw: bytes):
    value, _, state = budget._validate_document(_decode(raw), current=False)
    _require(state.version == 1 and value['scope'] == budget.scope)
    # Completed failed_or_uncertain attempts retain their full original charge.
    # An outstanding reservation is not converted into a completed attempt.
    _require(all(row['status'] != 'reserved' for row in value['attempts']))
    return value, state


def _plan(budget: PersistentBudget, source: bytes, candidate: bytes, pricing: dict) -> dict:
    lock = budget.lock_path.lstat()
    return dict(version=1, kind='v1-to-v2', authority=str(budget.authority_root),
                lock_identity=[lock.st_dev, lock.st_ino], source_sha256=_sha(source),
                candidate_sha256=_sha(candidate), pricing_sha256=_sha(_encode(pricing)))


def prepare_transition(
    budget: PersistentBudget, pricing_revision: dict, *, confirmed_billing_scope: str,
    approved_max_attempts: int, approved_max_reserved_usd: str,
    verified_pricing_sha256: str, writers_stopped: bool, dry_run: bool = True,
) -> TransitionPreview:
    """Validate first; dry-run writes nothing, prepare creates only private review artifacts."""
    with _operation(budget, writers_stopped):
        _require(type(dry_run) is bool)
        _require(confirmed_billing_scope == budget.billing_scope)
        _integer(approved_max_attempts, 1, 10000)
        _require(approved_max_attempts == budget.policy.max_attempts
                 and _money(approved_max_reserved_usd) == budget.policy.max_reserved_usd)
        _digest(verified_pricing_sha256)
        _require(_sha(_encode(pricing_revision)) == verified_pricing_sha256)
        raw = budget._read_bytes()
        legacy, state = _legacy_source(budget, raw)
        candidate = dict(version=2, scope=budget.scope, billing_scope=budget.billing_scope,
            provider='nebius', model=MODEL_ID, max_attempts=approved_max_attempts,
            max_reserved_usd=approved_max_reserved_usd, legacy=legacy,
            pricing_revisions=[pricing_revision], active_pricing=pricing_revision['id'], workflows=[])
        budget._validate_document(candidate)  # Including current <=24-hour pricing and lifetime caps.
        candidate_raw = _encode(candidate)
        plan = _plan(budget, raw, candidate_raw, pricing_revision)
        plan_raw = _encode(plan)
        plan_id = _sha(plan_raw)
        directory = None
        if not dry_run:
            directory = contained_path(budget.authority_root, f'.local/controlled-live/transition-{plan_id}')
            directory.mkdir(mode=0o700)  # Existing/partial preparations are not overwritten.
            _private_directory(budget, directory)
            budget._sync_directory()
            _write_private(budget, directory/'backup.v1.json', raw)
            _require(_sha(_read_private(budget, directory/'backup.v1.json')) == plan['source_sha256'])
            _write_private(budget, directory/'candidate.v2.json', candidate_raw)
            _write_private(budget, directory/'plan.json', plan_raw)
        return TransitionPreview(plan_id, plan['candidate_sha256'], plan['source_sha256'],
                                 state.attempts_consumed, str(state.reserved_usd), directory)


def _packet(budget: PersistentBudget, plan_id: str, *, current: bool):
    _digest(plan_id)
    directory = contained_path(budget.authority_root, f'.local/controlled-live/transition-{plan_id}')
    _private_directory(budget, directory)
    plan_raw = _read_private(budget, directory/'plan.json')
    _require(_sha(plan_raw) == plan_id)
    plan = _decode(plan_raw)
    _fields(plan, 'version kind authority lock_identity source_sha256 candidate_sha256 pricing_sha256')
    source = _read_private(budget, directory/'backup.v1.json')
    candidate_raw = _read_private(budget, directory/'candidate.v2.json')
    legacy, _ = _legacy_source(budget, source)
    candidate, _, state = budget._validate_document(_decode(candidate_raw), current=current)
    _require(state.version == 2 and candidate['legacy'] == legacy and candidate['workflows'] == []
             and len(candidate['pricing_revisions']) == 1)
    _require(plan == _plan(budget, source, candidate_raw, candidate['pricing_revisions'][0]))
    return directory, source, candidate_raw


def apply_transition(budget: PersistentBudget, plan_id: str, *, approved_plan_sha256: str,
                     writers_stopped: bool) -> dict:
    """Separate explicit approval of the exact prepared bytes and authority identity."""
    with _operation(budget, writers_stopped):
        _digest(plan_id)
        _require(approved_plan_sha256 == plan_id)
        directory, source, candidate_raw = _packet(budget, plan_id, current=True)
        _require(budget._read_bytes() == source)
        _legacy_source(budget, source)
        marker = (plan_id+'\n').encode()
        _require(not os.path.lexists(directory/'apply.started'))

        def verify():
            _require(budget._read_bytes() == candidate_raw)
            budget._read()  # Same schema, identity, pricing and counts after replacement.
            _write_private(budget, directory/'apply.completed', marker)

        budget._settle(candidate_raw, verify=verify,
            before_replace=lambda: _write_private(budget, directory/'apply.started', marker))
        return dict(action='applied', plan_sha256=plan_id, ledger_sha256=_sha(candidate_raw))


def rollback_transition(budget: PersistentBudget, plan_id: str, *, approved_rollback_sha256: str,
                        writers_stopped: bool) -> dict:
    """Explicit rollback of the exact untouched candidate, never of spending history."""
    with _operation(budget, writers_stopped):
        _digest(plan_id)
        rollback_id = _sha(('rollback:'+plan_id).encode())
        _require(approved_rollback_sha256 == rollback_id)
        # Expired *new* pricing is not permission to erase spending, but need not
        # prevent restoring an untouched v1 backup while all writers are stopped.
        directory, source, candidate_raw = _packet(budget, plan_id, current=False)
        _require(_read_private(budget, directory/'apply.completed') == (plan_id+'\n').encode())
        _require(budget._read_bytes() == candidate_raw)
        value, _, _ = budget._read(current=False)
        _require(value['version'] == 2 and value['workflows'] == [])
        marker = (rollback_id+'\n').encode()
        _require(not os.path.lexists(directory/'rollback.started'))

        def verify():
            _require(budget._read_bytes() == source)
            _require(budget._read(current=False)[2].version == 1)
            _write_private(budget, directory/'rollback.completed', marker)

        budget._settle(source, verify=verify,
            before_replace=lambda: _write_private(budget, directory/'rollback.started', marker))
        return dict(action='rolled_back', plan_sha256=plan_id, ledger_sha256=_sha(source))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Offline operator-controlled existing-ledger transition; no initialization.')
    parser.add_argument('--root', required=True, help='authorized application checkout')
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('prepare', help='validate and privately prepare; never replace ledger')
    prepare.add_argument('--pricing-file', required=True, help='private JSON relative to application root')
    prepare.add_argument('--verified-pricing-sha256', required=True, help='independently reviewed canonical pricing JSON digest')
    prepare.add_argument('--confirm-billing-scope', required=True)
    prepare.add_argument('--approve-max-attempts', required=True, type=int)
    prepare.add_argument('--approve-max-reserved-usd', required=True)
    prepare.add_argument('--dry-run', action='store_true', help='no filesystem writes, including no backup')
    for command in (prepare, commands.add_parser('apply'), commands.add_parser('rollback')):
        command.add_argument('--all-writers-stopped', action='store_true', required=True,
                             help='operator attests all old/new writers stopped and restart disabled')
        if command is not prepare:
            command.add_argument('--plan', required=True, help='prepared plan SHA-256')
            command.add_argument('--approve', required=True, help='plan digest for apply; separate rollback digest for rollback')
    args = parser.parse_args(argv)
    try:
        budget = configured_budget(Path(args.root))
        if args.action == 'prepare':
            path = contained_path(budget.root, args.pricing_file)
            _require(path.is_relative_to(budget.root/'.local'))
            pricing = _decode(_read_private(budget, path, root=budget.root))
            preview = prepare_transition(budget, pricing, confirmed_billing_scope=args.confirm_billing_scope,
                approved_max_attempts=args.approve_max_attempts, approved_max_reserved_usd=args.approve_max_reserved_usd,
                verified_pricing_sha256=args.verified_pricing_sha256, writers_stopped=args.all_writers_stopped,
                dry_run=args.dry_run)
            result = asdict(preview)
            result['artifact_directory'] = None if preview.artifact_directory is None else str(preview.artifact_directory)
            result['rollback_approval_sha256'] = _sha(('rollback:'+preview.plan_sha256).encode())
        elif args.action == 'apply':
            result = apply_transition(budget, args.plan, approved_plan_sha256=args.approve,
                                      writers_stopped=args.all_writers_stopped)
        else:
            result = rollback_transition(budget, args.plan, approved_rollback_sha256=args.approve,
                                         writers_stopped=args.all_writers_stopped)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        print('Transition denied or uncertain. Keep writers stopped; review private state. No automatic recovery.')
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
