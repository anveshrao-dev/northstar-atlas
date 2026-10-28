"""Opt-in Nebius HTTP transport. Imports and construction perform no network IO."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping

import httpx

from northstar_atlas.llm import (GenerationRequest, GenerationResponse, MODEL_ID,
                                 QwenTokenBound, TokenUsage)
from northstar_atlas.provider_diagnostics import (ClassifiedProviderError, FailureCategory,
                                                 transport_observation)

ENDPOINT = "https://api.tokenfactory.nebius.com/v1"


class ProviderError(ClassifiedProviderError):
    """Allowlisted messages only; no raw remote body, request or credential."""


class _ObservedTransport(httpx.BaseTransport):
    """Observe dispatch, not permit consumption; never retain the HTTP request."""
    def __init__(self, transport: httpx.BaseTransport):
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        observation = transport_observation()
        if observation is not None:
            observation.attempted = True
        response = self._transport.handle_request(request)
        if observation is not None:
            observation.http_status = response.status_code
        return response

    def close(self):
        self._transport.close()


def _reject_secret(value: object, secret: str, depth: int = 0) -> None:
    """Inspect decoded JSON, including JSON structured content nested in strings."""
    if depth > 32:
        raise ValueError("excessive response nesting")
    if isinstance(value, str):
        if secret in value:
            raise ValueError("credential echo")
        # The outer chat response may contain a second JSON document as content.
        # Decode that document too: escaped Unicode must not bypass suppression.
        if value.lstrip().startswith(("{", "[", '"')):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                return
            _reject_secret(decoded, secret, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_secret(key, secret, depth + 1)
            _reject_secret(item, secret, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _reject_secret(item, secret, depth + 1)


@dataclass(frozen=True)
class NebiusSettings:
    api_key: str = field(repr=False)
    endpoint: str = ENDPOINT
    model: str = MODEL_ID

    def __post_init__(self):
        # Never send credentials to arbitrary hosts, redirects, URL parameters or paths.
        if self.endpoint.rstrip("/") != ENDPOINT:
            raise ProviderError("endpoint is outside the approved Nebius HTTPS endpoint")
        if not isinstance(self.api_key, str) or not re.fullmatch(r"[\x21-\x7e]{12,4096}", self.api_key):
            raise ProviderError("missing or malformed Nebius credential")
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{1,160}", self.model):
            raise ProviderError("invalid model configuration")


def load_settings(root: Path, *, environ: Mapping[str, str] | None = None) -> NebiusSettings:
    """Read only selected settings privately; process/file conflicts fail closed."""
    env = os.environ if environ is None else environ
    root = root.resolve(strict=True)
    path = root / ".env"
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ProviderError("credential file must be local and not a symlink")
    ignored = subprocess.run(["git", "check-ignore", "-q", "--", ".env"], cwd=root, capture_output=True)
    tracked = subprocess.run(["git", "ls-files", "-z", "--", ".env"], cwd=root, capture_output=True)
    if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout:
        raise ProviderError(".env must be ignored and untracked/unstaged before loading credentials")
    keys = ("NEBIUS_API_KEY", "LLM_MODEL", "LLM_BASE_URL")
    values = {}
    try:
        if path.exists():
            with path.open() as stream:
                for line in stream:
                    name, separator, value = line.strip().partition("=")
                    if name not in keys: continue
                    if not separator or name in values: raise ValueError
                    parts = shlex.split(value, comments=True, posix=True)
                    if len(parts) > 1: raise ValueError
                    values[name] = parts[0] if parts else ""
        for key in keys:
            if env.get(key):
                if values.get(key) and env[key] != values[key]: raise ValueError
                values[key] = env[key]
    except (OSError, UnicodeError, ValueError):
        raise ProviderError("malformed, duplicate or conflicting local LLM configuration") from None
    return NebiusSettings(values.get("NEBIUS_API_KEY", ""), values.get("LLM_BASE_URL") or ENDPOINT,
                          values.get("LLM_MODEL") or MODEL_ID)


class NebiusProvider:
    """Bounded HTTP attempts; no auto-retry of potentially billable timeouts.

    HTTPX timeouts bound individual connect/read/write/pool waits, not an absolute
    wall-clock deadline. The initial live smoke plan uses one attempt. Optional
    retries cover only HTTP 429/502/503/504 and must be included in a caller's cost budget.
    """

    def __init__(self, settings: NebiusSettings, *, allow_paid: bool = False,
                 timeout_seconds: float = 30, max_attempts: int = 1,
                 transport: httpx.BaseTransport | None = None,
                 sleep: Callable[[float], None] = time.sleep, requires_budget: bool = False):
        if type(max_attempts) is not int or not 1 <= max_attempts <= 2:
            raise ProviderError("max_attempts must be one or two")
        if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120:
            raise ProviderError("timeout must be between zero and 120 seconds")
        self._requires_budget = requires_budget or os.environ.get('NORTHSTAR_CONTROLLED_LIVE') == '1'
        if self._requires_budget and max_attempts != 1:
            raise ProviderError('controlled inference requires one transport attempt')
        self._settings = settings
        self._allow_paid = allow_paid
        self._attempts = max_attempts
        self._sleep = sleep
        observed = _ObservedTransport(transport if transport is not None else httpx.HTTPTransport(trust_env=False))
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_seconds, connect=min(timeout_seconds, 5)),
                                    transport=observed, follow_redirects=False, trust_env=False)

    def __enter__(self): return self

    def __exit__(self, *args): self._client.close()

    def _request(self, method: str, path: str, payload: dict | None = None) -> tuple[dict, int, float]:
        started = time.perf_counter()
        attempts = self._attempts if method == "POST" else 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(method, self._settings.endpoint.rstrip("/") + path,
                    headers={"Authorization": "Bearer " + self._settings.api_key}, json=payload)
            except httpx.TimeoutException:
                raise ProviderError("provider timeout; no automatic retry, usage may be unknown",
                                    category=FailureCategory.TIMEOUT) from None
            except httpx.ConnectError:
                raise ProviderError("provider transport failure; no automatic retry",
                                    category=FailureCategory.CONNECTION) from None
            except httpx.HTTPError:
                raise ProviderError("provider transport failure; no automatic retry",
                                    category=FailureCategory.TRANSPORT) from None
            if response.status_code in (429, 502, 503, 504) and attempt < attempts:
                delay = 1.0
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except ValueError:
                        try: delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            raise ProviderError("invalid Retry-After; no retry") from None
                if not math.isfinite(delay) or delay > 5:
                    raise ProviderError("provider retry delay exceeds local retry budget")
                self._sleep(max(0, delay))
                continue
            if response.status_code != 200:
                raise ProviderError(f"provider returned HTTP {response.status_code}; response body suppressed",
                                    category=FailureCategory.HTTP)
            try:
                if len(response.content) > 2_000_000 or self._settings.api_key in response.text:
                    raise ValueError
                value = response.json()
                if not isinstance(value, dict): raise ValueError
                _reject_secret(value, self._settings.api_key)
            except (ValueError, UnicodeError, RecursionError):
                raise ProviderError("invalid or unsafe provider response", category=FailureCategory.ENVELOPE) from None
            return value, attempt, time.perf_counter() - started
        raise ProviderError("retry budget exhausted")

    def preflight(self) -> dict:
        """Authenticated model listing only; no completion, billing or account changes."""
        value, attempts, elapsed = self._request("GET", "/models")
        models = value.get("data")
        if not isinstance(models, list) or any(not isinstance(m, dict) or not isinstance(m.get("id"), str) for m in models):
            raise ProviderError("invalid model-list response")
        return {"endpoint": self._settings.endpoint, "model": self._settings.model,
                "credential_configured": True, "model_listed": any(m["id"] == self._settings.model for m in models),
                "generation_verified": False, "structured_output_verified": False,
                "account_limits": "unknown; inspect console before live testing",
                "http_calls": attempts, "latency_seconds": round(elapsed, 3)}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if self._requires_budget:
            raise ProviderError('controlled inference requires an active workflow reservation')
        return self._generate(request)

    def generate_budgeted(self, request: GenerationRequest, reservation) -> GenerationResponse:
        from .live_budget import BudgetError, WorkflowReservation
        observation = transport_observation()
        if observation is not None:
            observation.attempted = False
        try:
            if not isinstance(reservation, WorkflowReservation) or self._attempts != 1:
                raise BudgetError('invalid permit')
            reservation.claim_transport(request)
        except BudgetError:
            raise ProviderError('workflow reservation is absent or invalid', category=FailureCategory.REQUEST) from None
        return self._generate(request)

    def _generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._allow_paid:
            raise ProviderError("paid inference is disabled; separate approval and explicit opt-in are required",
                                category=FailureCategory.REQUEST)
        if request.model != self._settings.model:
            raise ProviderError("request model differs from configured model", category=FailureCategory.REQUEST)
        try:
            bound = QwenTokenBound(request.model).request_bound(request.system, request.user, request.schema)
            if (any(type(n) is not int or n <= 0 for n in (request.input_token_bound,request.output_tokens,request.context_tokens))
                    or request.input_token_bound < bound + 512
                    or request.input_token_bound + request.output_tokens > request.context_tokens
                    or request.context_tokens > 262144):
                raise ValueError
        except (ValueError, TypeError, UnicodeError):
            raise ProviderError("invalid request or unverified model token budget", category=FailureCategory.REQUEST) from None
        payload = {"model": request.model, "messages": [{"role": "system", "content": request.system},
                    {"role": "user", "content": request.user}], "temperature": 0, "max_tokens": request.output_tokens,
                   "n": 1, "stream": False, "store": False,
                   "response_format": {"type": "json_schema", "json_schema": {
                       "name": "grounded_answer", "strict": True, "schema": request.schema}}}
        value, attempts, elapsed = self._request("POST", "/chat/completions", payload)
        try:
            if value["model"] != request.model or len(value["choices"]) != 1: raise ValueError
            choice = value["choices"][0]; message = choice["message"]
            if (choice["finish_reason"] != "stop" or message["role"] != "assistant"
                    or message.get("refusal") or message.get("tool_calls") or message.get("function_call")
                    or not isinstance(message["content"], str) or not message["content"].strip()):
                raise ValueError
        except (ValueError, TypeError, KeyError, IndexError):
            raise ProviderError("model response, finish state or usage failed validation",
                                category=FailureCategory.ENVELOPE) from None
        try:
            raw_usage = value.get("usage")
            usage = None
            if raw_usage is not None:
                nums = [raw_usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")]
                if any(type(n) is not int or n < 0 for n in nums) or nums[0] + nums[1] != nums[2]: raise ValueError
                if nums[0] > request.input_token_bound or nums[1] > request.output_tokens: raise ValueError
                usage = TokenUsage(*nums)
            return GenerationResponse(message["content"], value["model"], choice["finish_reason"], usage, elapsed, attempts)
        except (ValueError, TypeError, KeyError, IndexError):
            raise ProviderError("model response, finish state or usage failed validation",
                                category=FailureCategory.USAGE) from None
