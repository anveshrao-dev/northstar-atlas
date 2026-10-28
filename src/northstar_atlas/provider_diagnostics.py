"""Fixed private failure metadata; never inspect exception text or provider payloads."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import json


class FailureCategory(StrEnum):
    TIMEOUT = 'transport_timeout'
    CONNECTION = 'connection_error'
    TRANSPORT = 'transport_error'
    HTTP = 'http_rejection'
    ENVELOPE = 'response_envelope_validation'
    USAGE = 'usage_validation'
    REQUEST = 'request_validation'
    INTERNAL = 'unexpected_internal'


class ClassifiedProviderError(ValueError):
    def __init__(self, message: str, *, category: FailureCategory = FailureCategory.INTERNAL):
        super().__init__(message)
        self.category = category


@dataclass
class TransportObservation:
    # None means an adapter did not instrument transport; it does not mean no IO.
    attempted: bool | None = None
    http_status: int | None = None


_observation: ContextVar[TransportObservation | None] = ContextVar('provider_transport', default=None)


@contextmanager
def observe_transport():
    """Per-invocation state, never cached on a shared provider or UI resource."""
    observation = TransportObservation()
    token = _observation.set(observation)
    try:
        yield observation
    finally:
        _observation.reset(token)


def transport_observation() -> TransportObservation | None:
    return _observation.get()


def failure_bytes(error: BaseException, observation: TransportObservation, *,
                  stage: str, ordinal: int, started_at: str,
                  fallback: FailureCategory) -> bytes:
    """Serialize only allowlisted primitives, never an exception or request object."""
    category = error.category if isinstance(error, ClassifiedProviderError) else fallback
    if type(category) is not FailureCategory:
        category = FailureCategory.INTERNAL
    if stage not in ('router', 'grader', 'generator', 'semantic_checker') or type(ordinal) is not int or ordinal not in (0, 1, 2):
        raise ValueError('invalid diagnostic stage')
    attempted = observation.attempted if type(observation.attempted) is bool else None
    status = observation.http_status
    if attempted is not True or type(status) is not int or not 100 <= status <= 599:
        status = None
    raw = json.dumps(dict(version=1, category=category.value, stage=stage, ordinal=ordinal,
        transport_attempted=attempted, http_status=status,
        started_at=datetime.fromisoformat(started_at).isoformat()),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > 512:
        raise ValueError('diagnostic exceeds bound')
    return raw
