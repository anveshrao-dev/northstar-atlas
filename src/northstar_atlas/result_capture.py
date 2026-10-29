"""Private result retention for the authorized one-shot Standard RAG runner.

The caller owns the durable budget reservation and passes an ask callable that
uses it. This module never reserves, settles, retries, or calls a provider twice.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Callable, get_args

from .evidence import CitationReference
from .ingestion.document_loader import APPROVED, DRAFT
from .qa import ErrorCode, PublicAnswer, QAFailure, QASuccess

_SCHEMA_VERSION = 1
_MAX_RECORD_BYTES = 32_768
_MAX_ANSWER_CHARS = 8_000
_MAX_CITATIONS = 40
_INVOCATION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_STAGES = {
    "input", "configuration", "retrieval", "context", "provider",
    "validation", "complete",
}


class ResultRecordingError(RuntimeError):
    """A bounded private result record could not be safely persisted."""


def result_path(root: Path, invocation_id: str) -> Path:
    """Create/validate the private result directory and return a unique target."""
    if not isinstance(invocation_id, str) or not _INVOCATION_ID.fullmatch(invocation_id):
        raise ValueError("invalid invocation identifier")
    if not root.is_absolute() or root != root.resolve(strict=True):
        raise ResultRecordingError("result root must be a canonical absolute directory")
    current = root
    for component in (".local", "linear-demo-results"):
        current = current / component
        try:
            current.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = current.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) & (0o022 if component == ".local" else 0o077)):
            raise ResultRecordingError("private result directory is unsafe")
    return current / f"{invocation_id}.json"


def _usage_record(diagnostics) -> dict:
    usage = diagnostics.usage
    if usage is None:
        return {}
    values = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    if any(type(value) is not int or value < 0 for value in values.values()):
        return {}
    return values


def _diagnostics_record(diagnostics) -> dict:
    stage = diagnostics.stage if diagnostics.stage in _STAGES else "unknown"
    provider_calls = diagnostics.provider_calls
    if type(provider_calls) is not int or provider_calls < 0:
        provider_calls = None
    return {
        "stage": stage,
        "backend": diagnostics.backend if diagnostics.backend in ("pinecone", "chroma", "custom") else "unknown",
        "mode": diagnostics.mode if diagnostics.mode in ("approved_only", "draft_review") else None,
        "provider_calls_reported": provider_calls,
        "token_usage": _usage_record(diagnostics),
    }


def _citation_record(citation: CitationReference) -> dict:
    if not isinstance(citation, CitationReference) or citation.status not in (APPROVED, DRAFT):
        raise ResultRecordingError("QA result contains an invalid citation")
    fields = (citation.document_id, citation.version, citation.section_id)
    if any(not isinstance(value, str) or not value or len(value) > 128 for value in fields):
        raise ResultRecordingError("QA result contains an invalid citation")
    return {
        "reference_id": citation.reference_id,
        "content_hash": citation.content_hash,
        "chunk_ids": citation.chunk_ids,
        "document_id": citation.document_id,
        "version": citation.version,
        "section_id": citation.section_id,
        "status": citation.status,
        "label": citation.label,
    }


def _record_from_result(result, *, store_answer: bool) -> dict:
    if type(store_answer) is not bool:
        raise TypeError("store_answer must be a boolean")

    if isinstance(result, QASuccess):
        public = result.public
        if not isinstance(public, PublicAnswer) or public.kind not in ("answer", "clarify", "abstain"):
            raise ResultRecordingError("QA success has an invalid public result")
        text = public.text
        if not isinstance(text, str) or len(text) > _MAX_ANSWER_CHARS:
            raise ResultRecordingError("QA answer exceeds the private record bound")
        if not isinstance(public.citations, tuple) or len(public.citations) > _MAX_CITATIONS:
            raise ResultRecordingError("QA result has too many citations")
        citations = tuple(_citation_record(c) for c in public.citations)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "status": "accepted",
            "answer_kind": public.kind,
            "answer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "answer_chars": len(text),
            "citations": citations,
            "diagnostics": _diagnostics_record(result.diagnostics),
        }
        if store_answer:
            record["answer_text"] = text
        return record

    if isinstance(result, QAFailure):
        public = result.public
        code = public.code if isinstance(public.code, str) else "unknown"
        if code not in get_args(ErrorCode):
            code = "unknown"
        return {
            "schema_version": _SCHEMA_VERSION,
            "status": "not_accepted",
            "failure_code": code,
            "diagnostics": _diagnostics_record(result.diagnostics),
        }

    raise ResultRecordingError("unexpected QA result type")


def _encode(record: dict) -> bytes:
    try:
        raw = json.dumps(record, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ResultRecordingError("QA result could not be safely serialized") from None
    if len(raw) > _MAX_RECORD_BYTES:
        raise ResultRecordingError("private QA result exceeds the record-size limit")
    return raw


def _claim(path: Path) -> tuple[int, os.stat_result]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o600:
        os.close(fd)
        raise ResultRecordingError("private result file permissions are unsafe")
    return fd, info


def _replace_claim(path: Path, claimed: os.stat_result, raw: bytes) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=".result-", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        current = path.lstat()
        if (not stat.S_ISREG(current.st_mode) or stat.S_ISLNK(current.st_mode)
                or current.st_dev != claimed.st_dev or current.st_ino != claimed.st_ino
                or current.st_uid != os.geteuid() or stat.S_IMODE(current.st_mode) != 0o600):
            raise ResultRecordingError("private result claim changed during execution")
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def capture_result(
    execute_once: Callable[[], QASuccess | QAFailure], *, root: Path, invocation_id: str,
) -> QASuccess | QAFailure:
    """Persist the typed public result before returning it for presentation.

    execute_once owns all authorization, reservation, QA invocation and finalization.
    No accounting is performed here. Exceptions propagate after a fixed safe record;
    an uncertain result file is retained and cannot authorize another execution.
    """
    path = result_path(root, invocation_id)
    claim_fd, claim_info = _claim(path)
    marker = _encode({"schema_version": _SCHEMA_VERSION, "status": "in_progress"})
    with os.fdopen(claim_fd, "wb") as stream:
        stream.write(marker)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    try:
        result = execute_once()
        record = _record_from_result(result, store_answer=True)
    except Exception:
        # No exception names/messages, rejected candidates, or provider payloads.
        _replace_claim(path, claim_info, _encode({
            "schema_version": _SCHEMA_VERSION, "status": "outcome_unknown",
        }))
        raise
    _replace_claim(path, claim_info, _encode(record))
    return result
