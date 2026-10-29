"""Shared, fail-closed local inference accounting. No initialization or migration.

POSIX locks coordinate cooperating processes on one durable filesystem, not
independent replicas. Estimates are conservative reservations, not billing.
Version 1 is read-only history. Version 2 must be provisioned in a separately
approved operation; this module cannot set pricing, increase limits or reset it.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from threading import get_ident
from types import MappingProxyType
from typing import Callable
from urllib.parse import urlsplit

from .llm import MODEL_ID, GenerationRequest, GenerationResponse, QwenTokenBound, TokenUsage, canonical_json
from .project import contained_path, validate_project_root
from .provider_diagnostics import FailureCategory, failure_bytes, observe_transport

LINEAR = 'linear_v1'
GRAPH = 'llm_routing_grading_semantic_v1'
UNIT = Decimal('0.000001')
MAX_ATTEMPTS = 3
MAX_COST = Decimal('0.100000')
_AUTHORITY_ENV = ('NORTHSTAR_BUDGET_AUTHORITY_ROOT', 'NORTHSTAR_BUDGET_AUTHORITY_ROOT_CONFIRMED')
_SLOTS = MappingProxyType({
    LINEAR: (('generator', 0, 30720, 2048),),
    GRAPH: (('router', 0, 7936, 256), ('grader', 1, 28672, 4096),
            ('grader', 2, 28672, 4096), ('generator', 0, 30720, 2048),
            ('semantic_checker', 1, 28672, 4096), ('generator', 1, 30720, 2048),
            ('semantic_checker', 2, 28672, 4096)),
})


class BudgetError(ValueError):
    """Fixed safe message: never include ledger, credential or question content."""


def _require(condition):
    if not condition:
        raise BudgetError('budget unavailable or reservation invalid; inference denied')


def _now():
    return datetime.now(timezone.utc)


def _fields(value, fields):
    _require(isinstance(value, dict) and set(value) == set(fields.split()))


def _integer(value, minimum=0, maximum=262144):
    _require(type(value) is int and minimum <= value <= maximum)
    return value


def _money(value):
    _require(isinstance(value, str) and re.fullmatch(r'[0-9]{1,6}(?:\.[0-9]{1,9})?', value))
    return Decimal(value)


def _timestamp(value):
    try:
        result = datetime.fromisoformat(value)
        _require(result.tzinfo is not None)
        return result
    except (TypeError, ValueError):
        raise BudgetError('invalid budget timestamp') from None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def _pricing(price, now=None):
    _fields(price, 'input_per_million_usd output_per_million_usd source confirmed_at valid_until')
    _require(all(0 < _money(price[k]) <= 1000 for k in ('input_per_million_usd', 'output_per_million_usd')))
    url = urlsplit(price['source'])
    _require(url.scheme == 'https' and url.hostname in
             {'nebius.com', 'docs.nebius.com', 'tokenfactory.nebius.com'}
             and not any((url.username, url.password, url.query, url.fragment)))
    start, end = _timestamp(price['confirmed_at']), _timestamp(price['valid_until'])
    _require(start < end <= start + timedelta(hours=24))
    if now is not None:
        _require(start <= now < end)


def _estimate(price, inputs, outputs):
    _integer(inputs, 1); _integer(outputs, 1)
    return ((inputs * _money(price['input_per_million_usd']) +
             outputs * _money(price['output_per_million_usd'])) / 1_000_000).quantize(UNIT, rounding=ROUND_CEILING)


def _legacy(value):
    """Validate donor v1 without changing/repricing even expired history."""
    _fields(value, 'version scope model max_attempts max_reserved_usd pricing attempts')
    _require(type(value['version']) is int and value['version'] == 1
             and isinstance(value['scope'], str) and re.fullmatch('[0-9a-f]{32}', value['scope'])
             and value['model'] == MODEL_ID and type(value['max_attempts']) is int
             and value['max_attempts'] == 3 and value['max_reserved_usd'] == '0.100000'
             and isinstance(value['attempts'], list) and len(value['attempts']) <= 3)
    price = value['pricing']
    if price is not None:
        _pricing(price)
    else:
        _require(not value['attempts'])
    ids, total = set(), Decimal(0)
    for row in value['attempts']:
        _fields(row, 'id reserved_at finished_at input_tokens output_tokens reserved_usd status')
        _require(isinstance(row['id'], str) and re.fullmatch('[0-9a-f]{32}', row['id']) and row['id'] not in ids)
        ids.add(row['id'])
        start = _timestamp(row['reserved_at'])
        _require(_timestamp(price['confirmed_at']) <= start < _timestamp(price['valid_until']))
        _require(row['status'] in ('reserved', 'succeeded', 'failed_or_uncertain'))
        _require(row['finished_at'] is None if row['status'] == 'reserved'
                 else _timestamp(row['finished_at']) >= start)
        cost = _estimate(price, row['input_tokens'], row['output_tokens'])
        _require(_money(row['reserved_usd']) == cost)
        total += cost
    _require(total <= MAX_COST)
    return len(ids), total


def _slot(profile, stage, ordinal):
    _require(profile in _SLOTS and type(ordinal) is int)
    matches = [s for s in _SLOTS[profile] if s[:2] == (stage, ordinal)]
    _require(len(matches) == 1)
    return matches[0]


def _order(profile, calls, stage, ordinal):
    _slot(profile, stage, ordinal)
    _require(all(c['status'] == 'returned' for c in calls))
    keys = [(c['stage'], c['ordinal']) for c in calls]
    _require((stage, ordinal) not in keys)
    if profile == LINEAR:
        _require(not keys)
    elif stage == 'router':
        _require(not keys)
    elif stage == 'grader':
        _require(bool(keys) and keys[-1][0] in ('router', 'grader')
                 and (keys[-1][0] != 'grader' or ordinal > keys[-1][1]))
    elif stage == 'generator':
        _require(bool(keys) and (keys[-1][0] == 'grader' if ordinal == 0
                                else keys[-1] == ('semantic_checker', 1)))
    else:
        _require(bool(keys) and keys[-1] == ('generator', ordinal-1))


@dataclass(frozen=True)
class BudgetPolicy:
    """Trusted operator policy, not UI input. Production defaults are unchanged."""
    max_attempts: int = MAX_ATTEMPTS
    max_reserved_usd: Decimal = MAX_COST

    def __post_init__(self):
        _integer(self.max_attempts, 1, 10000)
        _require(isinstance(self.max_reserved_usd, Decimal) and self.max_reserved_usd.is_finite()
                 and 0 < self.max_reserved_usd <= MAX_COST)


@dataclass(frozen=True)
class BudgetState:
    version: int
    attempts_consumed: int
    reserved_usd: Decimal


@dataclass(frozen=True)
class WorkflowRequest:
    invocation_id: str
    profile: str
    question: str
    mode: str
    backend: str
    reference_date: date

    def __post_init__(self):
        _require(isinstance(self.invocation_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', self.invocation_id)
                 and self.profile in _SLOTS and isinstance(self.question, str)
                 and 0 < len(self.question.strip()) <= 16000
                 and self.mode in ('approved_only', 'draft_review')
                 and self.backend in ('chroma', 'pinecone', 'custom')
                 and type(self.reference_date) is date)

    def binding(self, corpus_hash):
        return hashlib.sha256(canonical_json([self.profile, self.question, self.mode,
            self.backend, self.reference_date.isoformat(), corpus_hash, MODEL_ID]).encode()).hexdigest()


def _authority_root(application_root, requested, confirmed):
    """Trusted operator paths only; confirmation names one root, not a blanket opt-in."""
    try:
        if requested is None:
            _require(confirmed is None)
            return application_root
        raw = os.fspath(requested)
        _require(isinstance(raw, str) and bool(raw) and raw == raw.strip())
        path = Path(raw)
        # Reject aliases/traversal instead of normalizing them into authorization.
        _require(path.is_absolute() and '..' not in path.parts and str(path) == raw)
        if path != application_root or confirmed is not None:
            _require(confirmed is not None and os.fspath(confirmed) == raw)
        _require(path.resolve(strict=True) == path and path.is_dir())
        return path
    except (OSError, RuntimeError, TypeError, ValueError):
        raise BudgetError('budget authority is invalid or unauthorized; inference denied') from None


class PersistentBudget:
    def __init__(self, root: Path, *, scope: str | None = None, billing_scope: str | None = None,
                 policy: BudgetPolicy = BudgetPolicy(), clock=_now, before_request=None,
                 authority_root: Path | str | None = None,
                 authorized_authority_root: Path | str | None = None):
        # root always identifies application code, corpus, model and retrieval assets.
        # The separate authority controls only existing budget persistence files.
        self.root = validate_project_root(Path(root))
        self.authority_root = _authority_root(self.root, authority_root, authorized_authority_root)
        try:
            self.directory = contained_path(self.authority_root, '.local/controlled-live')
            self.path = contained_path(self.authority_root, '.local/controlled-live/budget.json')
            self.lock_path = contained_path(self.authority_root, '.local/controlled-live/budget.lock')
            self.pending_path = contained_path(self.authority_root, '.local/controlled-live/settlement.pending')
        except ValueError:
            raise BudgetError('budget authority paths are invalid; inference denied') from None
        self.scope, self.billing_scope, self.policy = scope, billing_scope, policy
        self.clock, self.before_request = clock, before_request
        self._blocked = False
        self._configured_authority = None
        self._path_identities = {}

    def _paths(self):
        if self._configured_authority is not None:
            _require(tuple(os.environ.get(name) for name in _AUTHORITY_ENV) == self._configured_authority)
        identities = {}
        for p in (self.authority_root, self.authority_root / '.local'):
            contained_path(self.authority_root, str(p.relative_to(self.authority_root)))
            info = p.lstat()
            _require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
                     and not stat.S_IMODE(info.st_mode) & 0o7022)
            identities[p] = (info.st_dev, info.st_ino)
        for p in (self.directory, self.path, self.lock_path):
            contained_path(self.authority_root, str(p.relative_to(self.authority_root)))
            info = p.lstat()
            kind = stat.S_ISDIR if p == self.directory else stat.S_ISREG
            _require(kind(info.st_mode) and info.st_uid == os.geteuid()
                     and stat.S_IMODE(info.st_mode) == (0o700 if p == self.directory else 0o600)
                     and (p == self.directory or info.st_nlink == 1))
            # Atomic ledger replacement is expected; replacing its lock/parents is not.
            if p != self.path: identities[p] = (info.st_dev, info.st_ino)
        _require(not self._path_identities or self._path_identities == identities)
        self._path_identities = identities

    def _opened_file(self, fd, path):
        info, current = os.fstat(fd), path.lstat()
        _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                 and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1
                 and (info.st_dev, info.st_ino) == (current.st_dev, current.st_ino))

    @contextmanager
    def _locked(self, *, diagnostic_approval=None):
        try:
            self._paths()
            fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
        except (OSError, ValueError):
            raise BudgetError('budget missing or inaccessible; inference denied') from None
        with os.fdopen(fd, 'r+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise BudgetError('budget is busy; no request was sent') from None
            try:
                self._paths()
                self._opened_file(lock.fileno(), self.lock_path)
                _require(not self._blocked and not os.path.lexists(self.pending_path))
                window = self.directory / 'diagnostic-window.json'
                plan = None
                if os.path.lexists(window) or diagnostic_approval is not None:
                    from .diagnostic_window import read_window
                    try:
                        plan = read_window(self, diagnostic_approval)
                    except (OSError, ValueError, TypeError, KeyError, RecursionError):
                        raise BudgetError('diagnostic window unavailable or unauthorized; inference denied') from None
                yield plan
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read_bytes(self):
        """Bounded, identity-checked read; callers already hold the existing lock."""
        try:
            self._paths()
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                self._opened_file(stream.fileno(), self.path)
                raw = stream.read(4_000_001)
            _require(len(raw) <= 4_000_000)
            return raw
        except (OSError, ValueError):
            raise BudgetError('budget missing or inaccessible; inference denied') from None

    def _read(self, *, current=True):
        try:
            value = json.loads(self._read_bytes(), object_pairs_hook=_unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
            return self._validate_document(value, current=current)
        except (ValueError, TypeError, RecursionError):
            raise BudgetError('budget missing, corrupt, incompatible or expired; inference denied') from None

    def _validate_document(self, value, *, current=True):
        """Same schema checks for stored history and an operator's unwritten candidate."""
        try:
            if isinstance(value, dict) and value.get('version') == 1:
                counts, cost = _legacy(value)
                _require(not current)  # Read-only compatibility; never silently migrate.
                return value, {}, BudgetState(1, counts, cost)
            fields = 'version scope billing_scope provider model max_attempts max_reserved_usd legacy pricing_revisions active_pricing workflows'
            if isinstance(value, dict) and 'diagnostic_request_sha256' in value:
                fields += ' diagnostic_request_sha256'
                _require(isinstance(value['diagnostic_request_sha256'], str)
                         and re.fullmatch('[0-9a-f]{64}', value['diagnostic_request_sha256']))
            _fields(value, fields)
            _require(type(value['version']) is int and value['version'] == 2
                     and isinstance(self.scope, str) and re.fullmatch('[0-9a-f]{32}', self.scope)
                     and isinstance(self.billing_scope, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', self.billing_scope)
                     and value['scope'] == self.scope and value['billing_scope'] == self.billing_scope
                     and value['provider'] == 'nebius' and value['model'] == MODEL_ID
                     and type(value['max_attempts']) is int and value['max_attempts'] == self.policy.max_attempts
                     and _money(value['max_reserved_usd']) == self.policy.max_reserved_usd)
            counts, cost = (0, Decimal(0)) if value['legacy'] is None else _legacy(value['legacy'])
            if value['legacy'] is not None:
                _require(value['legacy']['scope'] == value['scope'])
            _require(isinstance(value['pricing_revisions'], list) and 1 <= len(value['pricing_revisions']) <= 1000)
            prices = {}
            for row in value['pricing_revisions']:
                _fields(row, 'id input_per_million_usd output_per_million_usd source confirmed_at valid_until')
                key = row['id']
                _require(isinstance(key, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', key) and key not in prices)
                price = {k: v for k, v in row.items() if k != 'id'}
                _pricing(price); prices[key] = price
            _require(value['active_pricing'] in prices)
            if current:
                _pricing(prices[value['active_pricing']], self.clock())
            _require(isinstance(value['workflows'], list) and len(value['workflows']) <= 10000)
            ids = set()
            for flow in value['workflows']:
                _fields(flow, 'id binding profile mode backend price_id reserved_at finished_at status calls')
                _require(isinstance(flow['id'], str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}', flow['id'])
                         and flow['id'] not in ids and flow['profile'] in _SLOTS
                         and flow['mode'] in ('approved_only', 'draft_review')
                         and flow['backend'] in ('chroma', 'pinecone', 'custom')
                         and isinstance(flow['binding'], str) and re.fullmatch('[0-9a-f]{64}', flow['binding'])
                         and flow['price_id'] in prices and flow['status'] in ('reserved', 'closed', 'uncertain')
                         and isinstance(flow['calls'], list) and len(flow['calls']) <= len(_SLOTS[flow['profile']]))
                ids.add(flow['id']); price = prices[flow['price_id']]
                start = _timestamp(flow['reserved_at'])
                _require(_timestamp(price['confirmed_at']) <= start < _timestamp(price['valid_until']))
                _require(flow['finished_at'] is None if flow['status'] == 'reserved'
                         else _timestamp(flow['finished_at']) >= start)
                previous = []
                for call in flow['calls']:
                    _fields(call, 'stage ordinal input_tokens output_tokens status started_at finished_at usage')
                    _order(flow['profile'], previous, call['stage'], call['ordinal'])
                    _, _, inputs, outputs = _slot(flow['profile'], call['stage'], call['ordinal'])
                    _integer(call['input_tokens'], 1, inputs); _integer(call['output_tokens'], 1, outputs)
                    begun = _timestamp(call['started_at'])
                    _require(start <= begun < _timestamp(price['valid_until'])
                             and call['status'] in ('started', 'returned', 'failed_or_uncertain'))
                    _require(call['finished_at'] is None if call['status'] == 'started'
                             else _timestamp(call['finished_at']) >= begun)
                    _require(call['status'] != 'returned' or call['usage'] is not None)
                    if call['usage'] is not None:
                        _validate_usage(call['usage'], call['input_tokens'], call['output_tokens'])
                    previous.append(call)
                if flow['status'] == 'closed':
                    _require(all(c['status'] == 'returned' for c in previous))
                    selected = [_slot(flow['profile'], c['stage'], c['ordinal']) for c in previous]
                else:
                    selected = _SLOTS[flow['profile']]
                counts += len(selected)
                cost += sum((_estimate(price, s[2], s[3]) for s in selected), Decimal(0))
            _require(counts <= self.policy.max_attempts and cost <= self.policy.max_reserved_usd)
            return value, prices, BudgetState(2, counts, cost)
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
            raise BudgetError('budget missing, corrupt, incompatible or expired; inference denied') from None

    def _write(self, value):
        temporary = None
        try:
            self._paths()
            fd, name = tempfile.mkstemp(dir=self.directory, prefix='reservation-', suffix='.tmp')
            temporary = Path(name)
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                raw = value if isinstance(value, bytes) else json.dumps(
                    value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
                stream.write(raw)
                stream.flush(); os.fsync(stream.fileno())
            self._paths()
            os.replace(temporary, self.path)
            self._sync_directory()
        except (OSError, ValueError):
            raise BudgetError('budget could not be saved; outcome is uncertain') from None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def _sync_directory(self):
        directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _record_provider_failure(self, raw: bytes):
        """Latest failure only, under the held budget lock; never changes ledger data.

        A crash can leave one fixed pending file. Do not overwrite it or create
        more files: diagnostics then need operator review. Failure to save these
        optional diagnostics must not mask the original failure or skip accounting.
        """
        temporary = self.directory / 'provider-failure.pending'
        target = self.directory / 'provider-failure.json'
        created = False
        try:
            self._paths()
            _require(type(raw) is bytes and len(raw) <= 512)
            if os.path.lexists(target):
                info = target.lstat()
                _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
                         and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1)
                # NONBLOCK also prevents a raced FIFO replacement from hanging settlement.
                with os.fdopen(os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as stream:
                    self._opened_file(stream.fileno(), target)
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            created = True
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            self._paths()
            os.replace(temporary, target)
            self._sync_directory()
        finally:
            if created and os.path.lexists(temporary):
                temporary.unlink()

    def _mark_settlement(self):
        """A durable guard prevents an ambiguous closed write from releasing funds."""
        self._paths()
        contained_path(self.authority_root, str(self.pending_path.relative_to(self.authority_root)))
        fd = os.open(self.pending_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        self._sync_directory()

    def _settle(self, value, *, verify: Callable[[], None] | None = None,
                before_replace: Callable[[], None] | None = None):
        # Until both the ledger commit and guard removal are durable, no other
        # process may authorize a request. Never auto-recover this marker.
        try:
            self._mark_settlement()
            if before_replace is not None:
                before_replace()  # Transition receipts are protected by the same guard.
            self._write(value)
            if verify is not None:
                verify()  # Transition readback must succeed before releasing the guard.
            self.pending_path.unlink()
            self._sync_directory()
        except (OSError, ValueError):
            self._blocked = True
            if not os.path.lexists(self.pending_path):
                # A failed directory sync after unlink must not make a new
                # process mistake uncertain cleanup for successful settlement.
                try:
                    self._mark_settlement()
                except (OSError, ValueError):
                    pass  # Storage has failed; this process stays blocked.
            raise BudgetError('settlement is uncertain; reviewed recovery is required') from None

    def inspect_history(self, *, diagnostic_approval=None):
        with self._locked(diagnostic_approval=diagnostic_approval):
            return self._read(current=False)[2]

    def check_ready(self, *, profile=None):
        with self._locked():
            value, prices, state = self._read()
            _require('diagnostic_request_sha256' not in value)
            _require(state.attempts_consumed < self.policy.max_attempts and state.reserved_usd < self.policy.max_reserved_usd)
            if profile is not None:
                _require(profile in _SLOTS)
                slots = _SLOTS[profile]
                price = prices[value['active_pricing']]
                cost = sum((_estimate(price, s[2], s[3]) for s in slots), Decimal(0))
                _require(state.attempts_consumed + len(slots) <= self.policy.max_attempts
                         and state.reserved_usd + cost <= self.policy.max_reserved_usd)
            return state

    @contextmanager
    def reserve(self, request: WorkflowRequest, *, diagnostic_approval=None):
        _require(isinstance(request, WorkflowRequest))
        validate_project_root(self.root)  # Manifest and source hashes are checked again.
        with self._locked(diagnostic_approval=diagnostic_approval) as diagnostic_plan:
            validate_project_root(self.root)
            corpus = hashlib.sha256(contained_path(self.root, 'data/corpus_manifest.json').read_bytes()).hexdigest()
            value, prices, state = self._read()
            _require(all(f['id'] != request.invocation_id for f in value['workflows']))
            # Removing the private window must never reopen ordinary spending.
            _require('diagnostic_request_sha256' not in value or diagnostic_plan is not None)
            if diagnostic_plan is not None:
                from .diagnostic_window import check_request
                check_request(self, diagnostic_plan, request, corpus, value)
            price = prices[value['active_pricing']]
            slots = _SLOTS[request.profile]
            cost = sum((_estimate(price, s[2], s[3]) for s in slots), Decimal(0))
            _require(state.attempts_consumed + len(slots) <= self.policy.max_attempts
                     and state.reserved_usd + cost <= self.policy.max_reserved_usd)
            flow = dict(id=request.invocation_id, binding=request.binding(corpus), profile=request.profile,
                        mode=request.mode, backend=request.backend, price_id=value['active_pricing'],
                        reserved_at=self.clock().isoformat(), finished_at=None, status='reserved', calls=[])
            value['workflows'].append(flow)
            self._write(value)  # Complete envelope durable BEFORE resources/transport.
            reservation = WorkflowReservation(self, value, flow, price, corpus, cost)
            try:
                yield reservation
            finally:
                reservation._active = None
                reservation._open = False
                if not reservation._poisoned:
                    flow['status'] = 'uncertain' if any(c['status'] != 'returned' for c in flow['calls']) else 'closed'
                    flow['finished_at'] = self.clock().isoformat()
                    self._settle(value)
                else:
                    raise BudgetError('reservation persistence is uncertain; no response authorized') from None


def _validate_usage(usage, inputs, outputs):
    _fields(usage, 'prompt_tokens completion_tokens total_tokens')
    _integer(usage['prompt_tokens'], 0, inputs); _integer(usage['completion_tokens'], 0, outputs)
    _integer(usage['total_tokens'], 0, inputs+outputs)
    _require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'])


class WorkflowReservation:
    """Invocation-local permit. Never cached on a shared service or provider."""
    def __init__(self, budget, value, flow, price, corpus, cost):
        self._budget, self._value, self._flow, self._price = budget, value, flow, price
        self._corpus, self.reserved_usd = corpus, cost
        self._thread, self._open, self._poisoned, self._active = get_ident(), True, False, None
        self._transport_claimed = False

    @property
    def envelope(self):
        slots = _SLOTS[self._flow['profile']]
        return len(slots), sum(s[2] for s in slots), sum(s[3] for s in slots)

    def check_request(self, *, profile, question, mode, backend, reference_date):
        _require(self._open and not self._poisoned and self._thread == get_ident())
        spec = WorkflowRequest(self._flow['id'], profile, question, mode, backend, reference_date)
        _require(spec.binding(self._corpus) == self._flow['binding'])

    def assert_active(self, request):
        _require(self._open and not self._poisoned and get_ident() == self._thread and self._active is request)

    def claim_transport(self, request):
        """Consume the active stage's single transport permit, without another charge."""
        self.assert_active(request)
        _require(not self._transport_claimed)
        self._transport_claimed = True

    def check_coordinator(self, budget):
        _require(self._open and not self._poisoned and self._thread == get_ident() and self._budget is budget)

    def check_setup(self, budget, *, profile, backend):
        """Validate a live resource opener under the already-held workflow lock."""
        self.check_coordinator(budget)
        budget._paths()
        _require(self._flow['profile'] == profile and self._flow['backend'] == backend)
        _pricing(self._price, self._budget.clock())

    def _save(self):
        try: self._budget._write(self._value)
        except BaseException:
            self._poisoned = True
            raise

    def invoke(self, provider, stage, ordinal, request):
        _require(self._open and not self._poisoned and self._active is None and get_ident() == self._thread)
        _order(self._flow['profile'], self._flow['calls'], stage, ordinal)
        _, _, inputs, outputs = _slot(self._flow['profile'], stage, ordinal)
        _require(isinstance(request, GenerationRequest) and request.model == MODEL_ID)
        _integer(request.input_token_bound, 1, inputs); _integer(request.output_tokens, 1, outputs)
        _integer(request.context_tokens, 1, inputs+outputs)
        _require(request.input_token_bound + request.output_tokens <= request.context_tokens
                 and request.input_token_bound >= QwenTokenBound().request_bound(request.system, request.user, request.schema)+512
                 and getattr(provider, '_attempts', 1) == 1)
        _pricing(self._price, self._budget.clock())
        if self._budget.before_request is not None:
            self._budget.before_request()
        _pricing(self._price, self._budget.clock())
        call = dict(stage=stage, ordinal=ordinal, input_tokens=request.input_token_bound,
                    output_tokens=request.output_tokens, status='started', started_at=self._budget.clock().isoformat(),
                    finished_at=None, usage=None)
        self._flow['calls'].append(call); self._save()
        self._active = request
        self._transport_claimed = False
        with observe_transport() as observation:
            category = FailureCategory.INTERNAL
            try:
                guarded = getattr(provider, 'generate_budgeted', None)
                response = guarded(request, self) if callable(guarded) else provider.generate(request)
                category = FailureCategory.ENVELOPE
                _require(isinstance(response, GenerationResponse) and response.model == MODEL_ID
                         and type(response.attempts) is int and response.attempts == 1)
                category = FailureCategory.USAGE
                _require(isinstance(response.usage, TokenUsage))
                usage = {k: getattr(response.usage, k) for k in ('prompt_tokens','completion_tokens','total_tokens')}
                _validate_usage(usage, request.input_token_bound, request.output_tokens)
                call['usage'] = usage
                call['status'] = 'returned'
                return response
            except BaseException as error:
                call['status'] = 'failed_or_uncertain'
                try:
                    self._budget._record_provider_failure(failure_bytes(error, observation,
                        stage=stage, ordinal=ordinal, started_at=call['started_at'],
                        fallback=category if isinstance(error, BudgetError) else FailureCategory.INTERNAL))
                except (OSError, ValueError):
                    pass  # Optional diagnostics cannot authorize retries or bypass settlement.
                raise
            finally:
                self._active = None
                call['finished_at'] = self._budget.clock().isoformat()
                self._save()


class StageProvider:
    def __init__(self, reservation, provider, stage='generator', ordinal=0):
        self.reservation, self.provider, self.stage, self.ordinal = reservation, provider, stage, ordinal

    def generate(self, request):
        return self.reservation.invoke(self.provider, self.stage, self.ordinal, request)


def configured_budget(root):
    """Trusted server settings only; does not read credentials or create files."""
    from .live_preflight import require_private_preview
    # Operator policy must independently match the existing ledger. Never infer
    # permission from ledger contents or raise limits/create history automatically.
    attempts = os.environ.get('NORTHSTAR_BUDGET_MAX_ATTEMPTS', str(MAX_ATTEMPTS))
    _require(isinstance(attempts, str) and re.fullmatch(r'[1-9][0-9]{0,4}', attempts))
    policy = BudgetPolicy(max_attempts=int(attempts))
    authority = tuple(os.environ.get(name) for name in _AUTHORITY_ENV)
    budget = PersistentBudget(root, scope=os.environ.get('NORTHSTAR_BUDGET_SCOPE'),
                              billing_scope=os.environ.get('NORTHSTAR_BILLING_SCOPE'),
                              policy=policy, before_request=require_private_preview,
                              authority_root=authority[0], authorized_authority_root=authority[1])
    # Cached resources must fail closed if the operator changes/revokes selection.
    budget._configured_authority = authority
    return budget
