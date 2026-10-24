"""Exercise the evaluation validator CLI without touching the source corpus."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_evaluations.py"


class EvaluationValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary_root = ROOT / ".local/tmp"
        if not temporary_root.resolve().is_relative_to(ROOT):
            self.fail("Test temporary directory escapes the project")
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for folder in ("evaluations", "scripts", "data/policies"):
            (self.root / folder).mkdir(parents=True)
        shutil.copyfile(ROOT / "data/corpus_manifest.json", self.root / "data/corpus_manifest.json")
        manifest = json.loads((self.root / "data/corpus_manifest.json").read_text())
        for policy in manifest["policies"]:
            shutil.copyfile(ROOT / policy["filename"], self.root / policy["filename"])
        if VALIDATOR.exists():
            shutil.copyfile(VALIDATOR, self.root / "scripts/validate_evaluations.py")
        self.dataset = json.loads((ROOT / "evaluations/questions.json").read_text())
        self.write_dataset(self.dataset)

    def write_dataset(self, dataset: object) -> None:
        (self.root / "evaluations/questions.json").write_text(
            json.dumps(dataset, ensure_ascii=False), encoding="utf-8"
        )

    def run_validator(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(self.root / "scripts/validate_evaluations.py")],
            cwd=self.root / "evaluations",  # Must not depend on the caller's cwd.
            capture_output=True,
            text=True,
            check=False,
        )

    def assert_rejected(self, fragment: str) -> None:
        result = self.run_validator()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(fragment.lower(), result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)

    def test_current_dataset_passes_from_another_directory_without_writes(self) -> None:
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("15", result.stdout)
        self.assertEqual(result.stderr, "")
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_wrong_count_duplicate_ids_and_distribution_are_rejected(self) -> None:
        cases = []
        data = copy.deepcopy(self.dataset)
        data["questions"].pop()
        cases.append((data, "15"))
        data = copy.deepcopy(self.dataset)
        data["questions"][1]["question_id"] = data["questions"][0]["question_id"]
        cases.append((data, "duplicate question_id"))
        data = copy.deepcopy(self.dataset)
        data["questions"][3]["category"] = "C"
        cases.append((data, "distribution"))
        for data, message in cases:
            with self.subTest(message=message):
                self.write_dataset(data)
                self.assert_rejected(message)

    def test_bad_shapes_and_empty_rubric_fields_are_rejected(self) -> None:
        cases = [[], {**self.dataset, "questions": {}}]
        for field, value in [("required_evidence", []), ("required_qualifiers", []),
                             ("forbidden_inferences", [""]), ("notes", 7),
                             ("question", "  "), ("category", []),
                             ("required_evidence", [None])]:
            data = copy.deepcopy(self.dataset)
            data["questions"][0][field] = value
            cases.append(data)
        data = copy.deepcopy(self.dataset)
        del data["questions"][0]["expected_answer_summary"]
        cases.append(data)
        for data in cases:
            with self.subTest(data=str(data)[:80]):
                self.write_dataset(data)
                self.assert_rejected("error")

    def test_json_errors_duplicates_and_nonfinite_numbers_are_rejected(self) -> None:
        for raw in ('{broken', '{"questions": [], "questions": []}', '{"schema_version": NaN}'):
            with self.subTest(raw=raw):
                (self.root / "evaluations/questions.json").write_text(raw)
                self.assert_rejected("json")

    def test_unknown_document_or_section_is_rejected(self) -> None:
        for field, value, message in [("document_id", "FAKE-001", "document"),
                                      ("section_id", "99.1", "section")]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.dataset)
                data["questions"][0]["required_evidence"][0][field] = value
                self.write_dataset(data)
                self.assert_rejected(message)

    def test_modes_and_behaviors_are_validated_including_category_contract(self) -> None:
        for field, value in [("retrieval_mode", "all"), ("expected_behavior", "guess"),
                             ("retrieval_mode", "draft_review"), ("expected_behavior", "clarify")]:
            with self.subTest(field=field, value=value):
                data = copy.deepcopy(self.dataset)
                data["questions"][0][field] = value
                self.write_dataset(data)
                self.assert_rejected(field)

    def test_approved_only_rejects_draft_required_or_alternative_evidence(self) -> None:
        draft = {"document_id": "SEC-001", "section_id": "4.1"}
        for alternative in (False, True):
            with self.subTest(alternative=alternative):
                data = copy.deepcopy(self.dataset)
                if alternative:
                    data["questions"][0]["acceptable_alternative_evidence"] = [
                        {"replaces": data["questions"][0]["required_evidence"][0], "evidence": draft}
                    ]
                else:
                    data["questions"][0]["required_evidence"] = [draft]
                self.write_dataset(data)
                self.assert_rejected("approved_only")

    def test_alternative_must_replace_a_required_unit_and_resolve(self) -> None:
        for replacement, alternative, message in [
            ({"document_id": "HR-001", "section_id": "5.2"},
             {"document_id": "HR-001", "section_id": "4.3"}, "replaces"),
            ({"document_id": "HR-001", "section_id": "4.2"},
             {"document_id": "HR-001", "section_id": "99"}, "section"),
        ]:
            with self.subTest(message=message):
                data = copy.deepcopy(self.dataset)
                data["questions"][0]["acceptable_alternative_evidence"] = [
                    {"replaces": replacement, "evidence": alternative}
                ]
                self.write_dataset(data)
                self.assert_rejected(message)

    def test_optional_alternatives_can_be_omitted_or_structurally_valid(self) -> None:
        for question in self.dataset["questions"]:
            question.pop("acceptable_alternative_evidence")
        self.dataset["questions"][4]["acceptable_alternative_evidence"] = [
            {"replaces": {"document_id": "PLAT-001", "section_id": "4.3"},
             "evidence": {"document_id": "SRE-001", "section_id": "10.1"}}
        ]
        self.write_dataset(self.dataset)
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")

    def test_cross_document_category_requires_multiple_documents(self) -> None:
        self.dataset["questions"][3]["required_evidence"] = [{"document_id": "HR-001", "section_id": "4.4"}]
        self.write_dataset(self.dataset)
        self.assert_rejected("multiple documents")

    def test_policy_tampering_is_rejected_before_using_labels(self) -> None:
        path = self.root / "data/policies/HR-001.md"
        path.write_bytes(path.read_bytes() + b"\nChanged policy\n")
        self.assert_rejected("hash mismatch")

    def test_manifest_pin_detects_a_changed_experiment_snapshot(self) -> None:
        path = self.root / "data/corpus_manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_rejected("manifest hash")

    def test_manifest_cannot_read_an_outside_policy_path(self) -> None:
        path = self.root / "data/corpus_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["policies"][0]["filename"] = "../outside.md"
        path.write_text(json.dumps(manifest))
        self.assert_rejected("path")

    def test_duplicate_section_ids_are_rejected_even_with_updated_hashes(self) -> None:
        policy = self.root / "data/policies/HR-001.md"
        policy.write_bytes(policy.read_bytes() + b"\n### 4.2 Duplicate\n")
        path = self.root / "data/corpus_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["policies"][0]["content_hash"] = hashlib.sha256(policy.read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest))
        self.dataset["corpus_manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write_dataset(self.dataset)
        self.assert_rejected("duplicate section")


if __name__ == "__main__":
    unittest.main()
