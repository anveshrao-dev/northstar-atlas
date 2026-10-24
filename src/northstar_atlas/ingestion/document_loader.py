"""Load manifest-listed Markdown policies without changing their content.

Every source is validated before approval filtering. A bad excluded draft still
invalidates the corpus snapshot; callers never receive a partially checked list.
"""

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Literal


RetrievalMode = Literal["approved_only", "draft_review"]
APPROVED = "Approved (Synthetic)"
DRAFT = "Draft — Proposed (Synthetic)"
MANIFEST_PATH = "data/corpus_manifest.json"
MANIFEST_FIELDS = {"schema_version", "hash_algorithm", "filename_base", "policies"}
ENTRY_FIELDS = {
    "document_id", "title", "filename", "department", "version", "status",
    "effective_date", "classification", "language", "content_hash",
}


class DocumentLoadError(ValueError):
    """The corpus cannot be loaded safely or consistently with its manifest."""


@dataclass(frozen=True)
class PolicyDocument:
    """One verified policy; source_path is relative to the supplied project root."""

    document_id: str
    title: str
    department: str
    version: str
    status: str
    effective_date: date | None
    classification: str
    language: str
    source_path: Path
    content_hash: str
    content: str
    organization: str
    policy_owner: str
    review_cycle: str
    effective_date_text: str


def _project_root(project_root: Path) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DocumentLoadError("project root is missing or cannot be resolved") from exc
    if not root.is_dir():
        raise DocumentLoadError("project root must be a directory")
    return root


def _resolve_file(root: Path, relative: str, label: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise DocumentLoadError(f"{label}: path must be relative and stay inside the root")
    try:
        resolved = (root / path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise DocumentLoadError(f"{label}: missing file") from exc
    except (OSError, RuntimeError) as exc:
        raise DocumentLoadError(f"{label}: path cannot be resolved") from exc
    if not resolved.is_relative_to(root):
        raise DocumentLoadError(f"{label}: path resolves outside the project root")
    if not resolved.is_file():
        raise DocumentLoadError(f"{label}: path is not a regular file")
    return resolved


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(f"{label}: file cannot be read") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError("nonfinite JSON number")


def _iso_date(value: Any, label: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise DocumentLoadError(f"{label}: expected an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DocumentLoadError(f"{label}: invalid calendar date") from exc


def load_manifest(project_root: Path) -> list[dict[str, Any]]:
    """Validate schema-v1 manifest metadata and return entries in source order.

    Entries remain JSON dictionaries at this input boundary. load_documents
    validates the source files and converts them into typed PolicyDocument values.
    The manifest determines membership; no document IDs or corpus size are fixed.
    """
    root = _project_root(project_root)
    path = _resolve_file(root, MANIFEST_PATH, "manifest")
    raw = _read_bytes(path, "manifest")
    try:
        manifest = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, UnicodeError) as exc:
        # Do not echo malformed input, which could contain unrelated information.
        raise DocumentLoadError("manifest: invalid UTF-8 JSON or duplicate JSON keys") from exc
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise DocumentLoadError("manifest: expected an object with schema-v1 fields")
    if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["hash_algorithm"] != "sha256"
            or manifest["filename_base"] != "repository_root"):
        raise DocumentLoadError("manifest: unsupported schema, hash algorithm, or path base")
    entries = manifest["policies"]
    if not isinstance(entries, list) or not entries:
        raise DocumentLoadError("manifest: policies must be a nonempty array")
    seen_ids: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        label = f"manifest entry {index}"
        if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
            raise DocumentLoadError(f"{label}: missing or unknown policy metadata fields")
        for field in ENTRY_FIELDS - {"effective_date"}:
            value = entry[field]
            if (not isinstance(value, str) or not value.strip() or value != value.strip()
                    or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise DocumentLoadError(f"{label}: {field} must be a nonempty single-line string")
        doc_id = entry["document_id"]
        if not re.fullmatch(r"[A-Z][A-Z0-9]*-\d{3}", doc_id):
            raise DocumentLoadError(f"{label}: invalid document_id format")
        if doc_id in seen_ids:
            raise DocumentLoadError(f"{doc_id}: duplicate document_id in manifest")
        seen_ids.add(doc_id)
        # This corpus convention excludes review/evaluation files even if mislisted.
        if entry["filename"] != f"data/policies/{doc_id}.md":
            raise DocumentLoadError(f"{doc_id}: policy path must be data/policies/{doc_id}.md")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["content_hash"]):
            raise DocumentLoadError(f"{doc_id}: content_hash must be a lowercase SHA-256 digest")
        if entry["status"] == APPROVED:
            _iso_date(entry["effective_date"], f"{doc_id} effective_date")
        elif entry["status"] == DRAFT:
            if entry["effective_date"] is not None:
                raise DocumentLoadError(f"{doc_id}: draft effective_date must be null")
        else:
            raise DocumentLoadError(f"{doc_id}: unsupported policy status")
    return entries


def _source_metadata(content: str, entry: dict[str, Any]) -> dict[str, str]:
    """Validate the current corpus's heading and bold-label metadata format."""
    doc_id = entry["document_id"]
    lines = re.split(r"(?m)^## ", content, maxsplit=1)[0].splitlines()
    if not lines or lines[0] != f"# {doc_id} — {entry['title']}":
        raise DocumentLoadError(f"{doc_id}: source title does not match the manifest")
    metadata: dict[str, str] = {}
    for line in lines[1:]:
        match = re.fullmatch(r"\*\*([^*]+):\*\*[ \t]*(.*)", line)
        if match:
            key, value = match.groups()
            if key in metadata:
                raise DocumentLoadError(f"{doc_id}: duplicate source metadata field")
            metadata[key] = value.strip()
    labels = {
        "Document ID": "document_id", "Department": "department", "Version": "version",
        "Status": "status", "Classification": "classification", "Language": "language",
    }
    for label, field in labels.items():
        if metadata.get(label) != entry[field]:
            raise DocumentLoadError(f"{doc_id}: source {label} is missing or inconsistent")
    for label in ("Organization", "Policy Owner", "Review Cycle", "Effective Date"):
        if not metadata.get(label):
            raise DocumentLoadError(f"{doc_id}: source {label} is missing")
    effective = metadata["Effective Date"]
    if entry["status"] == APPROVED:
        _iso_date(effective, f"{doc_id} source Effective Date")
        if effective != entry["effective_date"]:
            raise DocumentLoadError(f"{doc_id}: source Effective Date does not match manifest")
    elif effective not in {"Pending approval", "Not effective — pending review and approval"}:
        raise DocumentLoadError(f"{doc_id}: draft source Effective Date must remain pending")
    if "Draft Date" in metadata:
        _iso_date(metadata["Draft Date"], f"{doc_id} source Draft Date")
    return metadata


def load_documents(
    project_root: Path,
    *,
    mode: RetrievalMode = "approved_only",
    reference_date: date | None = None,
) -> list[PolicyDocument]:
    """Load a fully validated corpus, then select the requested status/date mode.

    approved_only includes approved policies effective on or before reference_date
    (inclusive). An omitted reference_date means today's UTC calendar date.
    draft_review includes both supported statuses, including future approvals.
    """
    if mode not in ("approved_only", "draft_review"):
        raise DocumentLoadError("unsupported retrieval mode; use approved_only or draft_review")
    if reference_date is not None and type(reference_date) is not date:
        raise DocumentLoadError("reference_date must be a datetime.date, not a string or datetime")
    as_of = reference_date if reference_date is not None else datetime.now(timezone.utc).date()
    root = _project_root(project_root)
    entries = load_manifest(root)
    documents: list[PolicyDocument] = []
    for entry in entries:
        doc_id = entry["document_id"]
        path = _resolve_file(root, entry["filename"], doc_id)
        raw = _read_bytes(path, doc_id)
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != entry["content_hash"]:
            raise DocumentLoadError(f"{doc_id}: content hash mismatch; review the source and manifest")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentLoadError(f"{doc_id}: source is not valid UTF-8") from exc
        metadata = _source_metadata(content, entry)
        effective = (_iso_date(entry["effective_date"], f"{doc_id} effective_date")
                     if entry["effective_date"] is not None else None)
        documents.append(PolicyDocument(
            document_id=doc_id, title=entry["title"], department=entry["department"],
            version=entry["version"], status=entry["status"], effective_date=effective,
            classification=entry["classification"], language=entry["language"],
            source_path=Path(entry["filename"]), content_hash=actual_hash, content=content,
            organization=metadata["Organization"], policy_owner=metadata["Policy Owner"],
            review_cycle=metadata["Review Cycle"], effective_date_text=metadata["Effective Date"],
        ))
    if mode == "draft_review":
        return documents
    return [doc for doc in documents
            if doc.status == APPROVED and doc.effective_date is not None
            and doc.effective_date <= as_of]
