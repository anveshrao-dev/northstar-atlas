"""Validate the M1.8 dataset and its pinned policy corpus using only the stdlib.

Run with .venv/bin/python -B scripts/validate_evaluations.py.
This checks structure and provenance, not answer semantics or model quality.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


MANIFEST_PATH = "data/corpus_manifest.json"
CATEGORIES = {"A": 3, "B": 6, "C": 3, "D": 3}
CONTRACT = {
    "A": ("approved_only", "answer"),
    "B": ("draft_review", "answer"),
    "C": ("draft_review", "clarify"),
    "D": ("draft_review", "abstain"),
}
DOCUMENTS = {"HR-001", "IT-001", "PLAT-001", "SRE-001", "SEC-001", "COMP-001"}
TOP_FIELDS = {
    "schema_version", "dataset_id", "dataset_version", "corpus_manifest",
    "corpus_manifest_sha256", "questions",
}
QUESTION_FIELDS = {
    "question_id", "category", "question", "retrieval_mode", "expected_behavior",
    "expected_answer_summary", "required_evidence", "required_qualifiers",
    "forbidden_inferences", "notes",
}
OPTIONAL_FIELDS = {"acceptable_alternative_evidence"}


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently accepting the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_constant(value: str) -> Any:
    raise ValueError(f"nonfinite JSON number {value}")


def read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{path.name}: invalid JSON: {exc}") from exc
    return data, hashlib.sha256(raw).hexdigest()


def project_path(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"path escapes project: {relative}")
    return path


def load_corpus(root: Path) -> tuple[dict[str, dict[str, Any]], str]:
    manifest, manifest_hash = read_json(project_path(root, MANIFEST_PATH))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if (type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest.get("hash_algorithm") != "sha256"
            or manifest.get("filename_base") != "repository_root"):
        raise ValueError("unsupported manifest schema or hash/path convention")
    entries = manifest.get("policies")
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("manifest must list exactly six policies")
    corpus: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("manifest policy must be an object")
        doc = entry.get("document_id")
        if not isinstance(doc, str) or doc not in DOCUMENTS or doc in corpus:
            raise ValueError(f"unknown or duplicate manifest document_id: {doc!r}")
        filename = entry.get("filename")
        if filename != f"data/policies/{doc}.md":
            raise ValueError(f"{doc}: invalid policy path {filename!r}")
        raw = project_path(root, filename).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry.get("content_hash"):
            raise ValueError(f"{doc}: policy hash mismatch; stop and review the corpus")
        text = raw.decode("utf-8")
        metadata = dict(re.findall(r"^\*\*([^*]+):\*\* (.+?)\s*$", text, re.M))
        for key, label in [("document_id", "Document ID"), ("department", "Department"),
                           ("version", "Version"), ("status", "Status"),
                           ("classification", "Classification"), ("language", "Language")]:
            if not metadata.get(label) or metadata[label] != entry.get(key):
                raise ValueError(f"{doc}: manifest/source metadata mismatch for {key}")
        if not text.startswith(f"# {doc} — {entry.get('title')}\n"):
            raise ValueError(f"{doc}: manifest/source title mismatch")
        status = entry["status"]
        if status not in {"Approved (Synthetic)", "Draft — Proposed (Synthetic)"}:
            raise ValueError(f"{doc}: unsupported policy status")
        effective = metadata.get("Effective Date")
        if status == "Approved (Synthetic)":
            if not effective or entry.get("effective_date") != effective:
                raise ValueError(f"{doc}: effective date mismatch")
        elif (entry.get("effective_date") is not None
              or effective not in {"Pending approval", "Not effective — pending review and approval"}):
            raise ValueError(f"{doc}: draft effective date must remain pending/null")
        sections = re.findall(r"^#{2,3} (\d+(?:\.\d+)?)(?:\.| )", text, re.M)
        if not sections or len(sections) != len(set(sections)):
            raise ValueError(f"{doc}: missing or duplicate section identifiers")
        corpus[doc] = {"status": status, "sections": set(sections)}
    return corpus, manifest_hash


def validate_dataset(data: Any, corpus: dict[str, dict[str, Any]],
                     manifest_hash: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict) or set(data) != TOP_FIELDS:
        return ["dataset must be an object with exactly the documented top-level fields"]
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        errors.append("schema_version must be integer 1")
    for field in ("dataset_id", "dataset_version"):
        if not isinstance(data[field], str) or not data[field].strip():
            errors.append(f"{field} must be a nonempty string")
    if data["corpus_manifest"] != MANIFEST_PATH:
        errors.append(f"corpus_manifest must be {MANIFEST_PATH}")
    if data["corpus_manifest_sha256"] != manifest_hash:
        errors.append("manifest hash differs from the dataset pin; review labels before updating")
    questions = data["questions"]
    if not isinstance(questions, list):
        return errors + ["questions must be an array"]
    if len(questions) != 15:
        errors.append(f"expected exactly 15 questions, found {len(questions)}")
    seen_ids: set[str] = set()
    categories: Counter[str] = Counter()

    for index, question in enumerate(questions, start=1):
        context = f"questions[{index}]"
        if not isinstance(question, dict):
            errors.append(f"{context}: must be an object")
            continue
        missing = QUESTION_FIELDS - set(question)
        extra = set(question) - QUESTION_FIELDS - OPTIONAL_FIELDS
        if missing or extra:
            errors.append(f"{context}: missing fields {sorted(missing)}; unknown fields {sorted(extra)}")
        qid = question.get("question_id")
        if not isinstance(qid, str) or not re.fullmatch(r"EVAL-\d{3}", qid):
            errors.append(f"{context}: question_id must match EVAL-NNN")
        elif qid in seen_ids:
            errors.append(f"{context}: duplicate question_id {qid}")
        else:
            seen_ids.add(qid)
            context = qid
        for field in ("question", "expected_answer_summary", "notes"):
            value = question.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{context}: {field} must be a nonempty string")
        for field in ("required_qualifiers", "forbidden_inferences"):
            value = question.get(field)
            if (not isinstance(value, list) or not value
                    or any(not isinstance(item, str) or not item.strip() for item in value)):
                errors.append(f"{context}: {field} must be a nonempty array of nonempty strings")
        category = question.get("category")
        mode = question.get("retrieval_mode")
        behavior = question.get("expected_behavior")
        if not isinstance(mode, str) or mode not in {"approved_only", "draft_review"}:
            errors.append(f"{context}: invalid retrieval_mode")
        if not isinstance(behavior, str) or behavior not in {"answer", "clarify", "abstain"}:
            errors.append(f"{context}: invalid expected_behavior")
        if not isinstance(category, str) or category not in CONTRACT:
            errors.append(f"{context}: category must be A, B, C, or D")
        else:
            categories[category] += 1
            required_mode, required_behavior = CONTRACT[category]
            if mode != required_mode:
                errors.append(f"{context}: category {category} requires retrieval_mode {required_mode}")
            if behavior != required_behavior:
                errors.append(f"{context}: category {category} requires expected_behavior {required_behavior}")

        def evidence_ref(value: Any, field: str) -> tuple[str, str] | None:
            if not isinstance(value, dict) or set(value) != {"document_id", "section_id"}:
                errors.append(f"{context}: {field} must contain document_id and section_id")
                return None
            doc, section = value["document_id"], value["section_id"]
            if not isinstance(doc, str) or doc not in corpus:
                errors.append(f"{context}: {field} references unknown document {doc!r}")
                return None
            if not isinstance(section, str) or section not in corpus[doc]["sections"]:
                errors.append(f"{context}: {field} references unknown section {doc} {section!r}")
                return None
            if mode == "approved_only" and corpus[doc]["status"] != "Approved (Synthetic)":
                errors.append(f"{context}: approved_only cannot use draft evidence {doc} §{section}")
            return doc, section

        required = question.get("required_evidence")
        units: set[tuple[str, str]] = set()
        if not isinstance(required, list) or not required:
            errors.append(f"{context}: required_evidence must be a nonempty array")
        else:
            for value in required:
                ref = evidence_ref(value, "required_evidence")
                if ref in units:
                    errors.append(f"{context}: duplicate required evidence {ref}")
                if ref is not None:
                    units.add(ref)
        if category == "B" and len({doc for doc, _ in units}) < 2:
            errors.append(f"{context}: category B requires evidence from multiple documents")
        alternatives = question.get("acceptable_alternative_evidence", [])
        if not isinstance(alternatives, list):
            errors.append(f"{context}: acceptable_alternative_evidence must be an array")
        else:
            for alternative in alternatives:
                if not isinstance(alternative, dict) or set(alternative) != {"replaces", "evidence"}:
                    errors.append(f"{context}: alternative needs replaces and evidence objects")
                    continue
                replacement = evidence_ref(alternative["replaces"], "alternative replaces")
                evidence_ref(alternative["evidence"], "alternative evidence")
                if replacement not in units:
                    errors.append(f"{context}: alternative replaces a unit not in required_evidence")
    if dict(categories) != CATEGORIES:
        errors.append(f"category distribution must be {CATEGORIES}, found {dict(categories)}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        corpus, manifest_hash = load_corpus(root)
        data, _ = read_json(project_path(root, "evaluations/questions.json"))
        errors = validate_dataset(data, corpus, manifest_hash)
    except (OSError, ValueError, UnicodeError) as exc:
        errors = [str(exc)]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAIL: {len(errors)} validation error(s)", file=sys.stderr)
        return 1
    print("PASS: 15 unique questions; categories A/B/C/D = 3/6/3/3")
    print("PASS: source hashes, manifest pin, evidence sections, modes, and behaviors")
    print("Semantic answer review is separate; no retrieval or model evaluation was run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
