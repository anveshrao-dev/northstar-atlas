"""Explicit Pinecone control plane; imports never load credentials or access the cloud."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from urllib.parse import urlsplit
from uuid import UUID

from northstar_atlas.project import contained_path, validate_project_root

from .records import canonical_json



class CloudError(ValueError):
    """Allowlisted messages only; never wrap raw provider exception text."""


@dataclass(frozen=True)
class CloudSettings:
    api_key: str = field(repr=False)
    name: str
    cloud: str
    region: str
    organization: str
    project_name: str
    project_id: str
    plan: str

    def __post_init__(self):
        try:
            if (self.cloud, self.region, self.plan) != ("azure", "eastus2", "Builder"):
                raise ValueError
            if not isinstance(self.name, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,43}[a-z0-9])?", self.name):
                raise ValueError
            if not isinstance(self.project_id, str) or str(UUID(self.project_id)) != self.project_id:
                raise ValueError
            for value in (self.organization, self.project_name):
                if (not isinstance(value, str) or not value.strip() or value != value.strip()
                        or len(value) > 200 or any(ord(c) < 32 or ord(c) == 127 for c in value)):
                    raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise CloudError("invalid explicit Pinecone target identity or unsupported placement") from None
        if not isinstance(self.api_key, str) or not re.fullmatch(r"[A-Za-z0-9_-]{20,512}", self.api_key):
            raise CloudError("missing or malformed Pinecone credential")


def local_path(root: Path, relative: str) -> Path:
    """Reject symlinks at every component before reading/writing project artifacts."""
    try:
        return contained_path(root, relative)
    except ValueError:
        raise CloudError("artifact path is invalid or outside project") from None


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def load_settings(root: Path, *, environ=None) -> CloudSettings:
    """Use a complete environment only when no local credential file exists."""
    try:
        root = validate_project_root(root)
    except ValueError:
        raise CloudError("invalid or unauthorized project root/corpus") from None
    env = os.environ if environ is None else environ
    path = local_path(root, ".env")
    has_file = path.exists()
    if has_file:
        ignored = subprocess.run(["git", "check-ignore", "-q", "--", ".env"], cwd=root, capture_output=True)
        tracked = subprocess.run(["git", "ls-files", "-z", "--", ".env"], cwd=root, capture_output=True)
        if ignored.returncode != 0 or tracked.returncode != 0 or tracked.stdout:
            raise CloudError(".env must be ignored and untracked/unstaged before credentials are loaded")
    wanted = ("PINECONE_API_KEY", "PINECONE_INDEX_NAME", "PINECONE_CLOUD", "PINECONE_REGION")
    values = {}
    try:
        if has_file:
            for line in path.read_text().splitlines():
                name, separator, value = line.strip().partition("=")
                if name not in wanted: continue
                parts = shlex.split(value, comments=True, posix=True)
                if not separator or name in values or len(parts) != 1:
                    raise ValueError
                values[name] = parts[0]
        else:
            # Check completeness before accessing any credential value.
            if not all(name in env for name in wanted): raise ValueError
            values = {name: env[name] for name in wanted}
        if (set(values) != set(wanted)
                or any(not isinstance(value, str) or not value.strip() for value in values.values())):
            raise ValueError
        if any(name in env and env[name] != value for name, value in values.items()):
            raise ValueError
    except (OSError, UnicodeError, ValueError):
        raise CloudError("missing, malformed, duplicate or conflicting local Pinecone configuration") from None
    try:
        target = json.loads(local_path(root, ".local/pinecone/target.json").read_text(),
                            object_pairs_hook=_unique_object)
        fields = {"name", "cloud", "region", "organization", "project_name", "project_id", "plan"}
        if not isinstance(target, dict) or set(target) != fields:
            raise ValueError
        if any(values[env_key] != target[field] for env_key, field in
               (("PINECONE_INDEX_NAME", "name"), ("PINECONE_CLOUD", "cloud"), ("PINECONE_REGION", "region"))):
            raise ValueError
        return CloudSettings(api_key=values["PINECONE_API_KEY"], **target)
    except (OSError, UnicodeError, ValueError, TypeError):
        raise CloudError("explicit Pinecone target missing, malformed or conflicting; no account fallback") from None


def _confirmation(settings: CloudSettings) -> dict:
    return {"organization": settings.organization, "project_name": settings.project_name, "project_id": settings.project_id,
            "index_name": settings.name,
            "cloud": settings.cloud, "region": settings.region, "plan": settings.plan,
            "basis": "owner Console attestation; project-key API cannot independently verify identity",
            "key_digest": hashlib.sha256(settings.api_key.encode()).hexdigest()}


def record_confirmation(root: Path, settings: CloudSettings, *, project_id: str):
    """Called only after explicit owner confirmation; not exposed as an auto-confirm CLI."""
    if project_id != settings.project_id: raise CloudError("project identity differs from approved project")
    path = local_path(root, ".local/pinecone/target-confirmation.json")
    value = _confirmation(settings)
    if path.exists():
        check_confirmation(root, settings)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        stream.write(canonical_json({**value, "confirmed_at": datetime.now(timezone.utc).isoformat()}))


def check_confirmation(root: Path, settings: CloudSettings):
    try:
        value = json.loads(local_path(root, ".local/pinecone/target-confirmation.json").read_text())
        if any(value.get(k) != v for k, v in _confirmation(settings).items()): raise ValueError
    except (OSError, ValueError, AttributeError):
        raise CloudError("owner Console confirmation missing or differs for this credential/project/region; stop") from None


def safe_call(operation, function, **kwargs):
    try:
        return function(**kwargs)
    except Exception:
        raise CloudError(f"Pinecone {operation} failed; reconcile read-only before retrying") from None


def as_dict(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def validate_description(value, settings: CloudSettings, *, require_ready=False) -> dict:
    data = as_dict(value)
    try:
        host = data.get("host")
        url = urlsplit(host or "")
        host_valid = (url.scheme == "https" and url.hostname and url.hostname.endswith(".pinecone.io")
                      and not url.username and not url.password and not url.query and not url.fragment
                      and url.port in (None, 443) and url.path in ("", "/"))
        # SDK 10 permits no host while initialization is in progress. A host
        # that is present must still be trusted, even before readiness.
        host_pending = host is None and data["status"]["ready"] is False and not require_ready
        vector = data["schema"]["fields"]["embedding"]
        valid = (data["name"] == settings.name and data["deletion_protection"] == "enabled"
                 and data["deployment"]["deployment_type"] == "managed"
                 and data["deployment"]["cloud"] == settings.cloud and data["deployment"]["region"] == settings.region
                 and set(data["schema"]["fields"]) == {"embedding"}
                 and vector["type"] == "dense_vector" and type(vector["dimension"]) is int
                 and vector["dimension"] == 384 and vector["metric"] == "cosine" and not vector.get("embed")
                 and (host_valid or host_pending)
                 and (not require_ready or data["status"]["ready"] is True))
    except (KeyError, TypeError, AttributeError, ValueError): valid = False
    if not valid: raise CloudError("incompatible index configuration or host; no automatic changes")
    return data


def preflight(indexes, root: Path, settings: CloudSettings) -> dict:
    check_confirmation(root, settings)
    # Materialize inside safe_call so pagination errors cannot leak SDK details.
    found = safe_call("list indexes", lambda: [as_dict(i) for i in indexes.list()])
    existing = next((i for i in found if i["name"] == settings.name), None)
    if existing is not None:
        existing = validate_description(safe_call("describe", indexes.describe, name=settings.name), settings)
    return {"authentication": "succeeded", "project_id": settings.project_id,
            "project_verification": "owner Console attestation (not independently API verified)",
            "region_availability": "azure/eastus2 on Builder confirmed by owner",
            "index_state": "absent" if existing is None else "compatible",
            "configuration": {"name": settings.name, "cloud": settings.cloud, "region": settings.region,
                              "dimension": 384, "metric": "cosine", "integrated_inference": False},
            "description": existing, "credit_balance": "not checked; usage is not assumed free"}


def create_index(indexes, root: Path, settings: CloudSettings, *, attempts=60, interval=5) -> dict:
    if type(attempts) is not int or not 1 <= attempts <= 60 or not 0 <= interval <= 5:
        raise CloudError("invalid bounded readiness settings")
    report = preflight(indexes, root, settings)
    value = report["description"]
    if value is None:
        value = safe_call("create (outcome may be unknown)", indexes.create, name=settings.name,
            schema={"fields": {"embedding": {"type": "dense_vector", "dimension": 384, "metric": "cosine"}}},
            deployment={"deployment_type": "managed", "cloud": settings.cloud, "region": settings.region},
            read_capacity={"mode": "OnDemand"}, deletion_protection="enabled", timeout=-1)
    deadline = time.monotonic() + 300
    for attempt in range(attempts):
        data = validate_description(value, settings)
        if data["status"]["ready"] is True: return data
        if time.monotonic() >= deadline: break
        if attempt + 1 < attempts:
            time.sleep(interval)
            value = safe_call("readiness describe", indexes.describe, name=settings.name)
    raise CloudError("index readiness deadline exhausted; inspect status; do not recreate or delete")


@contextmanager
def sdk_client(settings: CloudSettings):
    """Pinned SDK, explicit official host, TLS, no debug/proxy/env header overrides."""
    from pinecone import Pinecone, RetryConfig
    names = ("PINECONE_DEBUG_CURL", "PINECONE_ADDITIONAL_HEADERS", "PINECONE_CONTROLLER_HOST",
             "PINECONE_GRPC_SCHEME", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
             "http_proxy", "https_proxy", "all_proxy")
    saved = {k: os.environ.pop(k) for k in names if k in os.environ}
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        # No automatic retry of ambiguous control-plane writes. Data-plane SDK
        # retains its finite default retries; the adapter adds no retry wrapper.
        with Pinecone(api_key=settings.api_key, host="https://api.pinecone.io", timeout=10,
                      ssl_verify=True, retry_config=RetryConfig(max_retries=0)) as client:
            yield client
    finally:
        logging.disable(previous)
        os.environ.update(saved)
