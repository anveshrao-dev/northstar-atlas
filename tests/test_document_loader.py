"""Document loading tests; failure fixtures stay inside the project's .local/."""

import copy
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 10, 22)


class DocumentLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("northstar_atlas.ingestion"),
            "The document-loading package has not been implemented",
        )
        self.loader = importlib.import_module("northstar_atlas.ingestion.document_loader")
        temporary_root = ROOT / ".local/tmp"
        self.assertTrue(temporary_root.resolve().is_relative_to(ROOT))
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "repo"
        (self.root / "data/policies").mkdir(parents=True)
        self.manifest_path = self.root / "data/corpus_manifest.json"
        shutil.copyfile(ROOT / "data/corpus_manifest.json", self.manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        for entry in self.manifest["policies"]:
            shutil.copyfile(ROOT / entry["filename"], self.root / entry["filename"])

    def load(self, mode: str = "draft_review", reference_date: date = REFERENCE_DATE):
        return self.loader.load_documents(
            self.root, mode=mode, reference_date=reference_date
        )

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def change_source(self, index: int, raw: bytes) -> None:
        entry = self.manifest["policies"][index]
        (self.root / entry["filename"]).write_bytes(raw)
        entry["content_hash"] = hashlib.sha256(raw).hexdigest()
        self.save_manifest()

    def test_loads_all_six_with_correct_hashes_and_unchanged_content(self) -> None:
        documents = self.load()
        self.assertEqual([d.document_id for d in documents],
                         ["HR-001", "IT-001", "PLAT-001", "SRE-001", "SEC-001", "COMP-001"])
        for doc, entry in zip(documents, self.manifest["policies"], strict=True):
            raw = (self.root / entry["filename"]).read_bytes()
            self.assertEqual(doc.content.encode("utf-8"), raw)
            self.assertEqual(doc.content_hash, hashlib.sha256(raw).hexdigest())
            self.assertEqual(doc.source_path, Path(entry["filename"]))
            for field in ("title", "department", "version", "status", "classification", "language"):
                self.assertEqual(getattr(doc, field), entry[field])
        self.assertEqual(documents[0].organization, "Northstar Technologies")
        self.assertEqual(documents[0].policy_owner, "Director of Human Resources")
        self.assertEqual(documents[0].review_cycle, "Annual")
        with self.assertRaises(FrozenInstanceError):
            documents[0].status = "Draft — Proposed (Synthetic)"

    def test_approved_only_loads_hr_at_reference_date_and_today(self) -> None:
        self.assertEqual([d.document_id for d in self.load("approved_only")], ["HR-001"])
        self.assertEqual(
            self.loader.load_documents(self.root),
            self.loader.load_documents(
                self.root, reference_date=datetime.now(timezone.utc).date()
            ),
        )
        self.assertEqual(self.load("approved_only")[0].effective_date, date(2026, 10, 1))

    def test_effective_date_boundary_and_future_approved_documents(self) -> None:
        self.assertEqual(self.load("approved_only", date(2026, 9, 30)), [])
        self.assertEqual(len(self.load("approved_only", date(2026, 10, 1))), 1)
        self.assertEqual(len(self.load("draft_review", date(2026, 9, 30))), 6)

    def test_filter_does_not_hardcode_approved_document_ids_or_corpus_count(self) -> None:
        entry = copy.deepcopy(self.manifest["policies"][0])
        raw = (self.root / entry["filename"]).read_bytes().replace(b"HR-001", b"ALT-002")
        entry.update(document_id="ALT-002", filename="data/policies/ALT-002.md",
                     content_hash=hashlib.sha256(raw).hexdigest())
        (self.root / entry["filename"]).write_bytes(raw)
        self.manifest["policies"] = [entry]
        self.save_manifest()
        self.assertEqual([d.document_id for d in self.load("approved_only")], ["ALT-002"])

    def test_pending_dates_remain_none_and_original_text_is_preserved(self) -> None:
        documents = self.load()
        self.assertTrue(all(d.effective_date is None for d in documents[1:]))
        self.assertEqual(documents[1].effective_date_text,
                         "Not effective — pending review and approval")
        self.assertEqual(documents[2].effective_date_text, "Pending approval")
        self.assertTrue(all(d.status == "Draft — Proposed (Synthetic)" for d in documents[1:]))

    def test_crlf_and_trailing_content_are_not_normalized(self) -> None:
        path = self.root / self.manifest["policies"][0]["filename"]
        raw = path.read_bytes().replace(b"\n", b"\r\n") + b"\r\n  \r\n"
        self.change_source(0, raw)
        self.assertEqual(self.load()[0].content.encode("utf-8"), raw)

    def test_modified_content_is_rejected_even_when_draft_would_be_filtered_out(self) -> None:
        path = self.root / self.manifest["policies"][1]["filename"]
        path.write_bytes(path.read_bytes() + b"\nChanged\n")
        for mode in ("approved_only", "draft_review"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                self.loader.DocumentLoadError, "IT-001.*hash mismatch"
            ):
                self.load(mode)

    def test_missing_policy_is_reported(self) -> None:
        (self.root / self.manifest["policies"][0]["filename"]).unlink()
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "HR-001.*missing"):
            self.load()

    def test_missing_manifest_is_reported(self) -> None:
        self.manifest_path.unlink()
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "manifest.*missing"):
            self.load()

    def test_malformed_json_duplicate_keys_and_nonfinite_values_are_rejected(self) -> None:
        for raw in ('{broken', '{"policies": [], "policies": []}', '{"schema_version": NaN}'):
            with self.subTest(raw=raw):
                self.manifest_path.write_text(raw)
                with self.assertRaisesRegex(self.loader.DocumentLoadError, "JSON"):
                    self.load()

    def test_invalid_manifest_structure_entries_status_and_dates_are_rejected(self) -> None:
        original = copy.deepcopy(self.manifest)
        mutations = [
            lambda m: m.update(schema_version=True),
            lambda m: m.update(policies=[]),
            lambda m: m["policies"].append(copy.deepcopy(m["policies"][0])),
            lambda m: m["policies"][0].pop("department"),
            lambda m: m["policies"][0].update(status="Approved"),
            lambda m: m["policies"][0].update(effective_date=None),
            lambda m: m["policies"][0].update(effective_date="2026-02-30"),
            lambda m: m["policies"][0].update(effective_date="20261001"),
            lambda m: m["policies"][1].update(effective_date="2026-10-20"),
            lambda m: m["policies"][0].update(version=1.0),
            lambda m: m["policies"][0].update(content_hash="invalid"),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.manifest = copy.deepcopy(original)
                mutate(self.manifest)
                self.save_manifest()
                with self.assertRaises(self.loader.DocumentLoadError):
                    self.load()
        for data in ([], {"policies": "invalid"}):
            self.manifest_path.write_text(json.dumps(data))
            with self.assertRaises(self.loader.DocumentLoadError):
                self.load()

    def test_parent_traversal_and_absolute_paths_are_rejected(self) -> None:
        for filename in ("../outside.md", "/etc/passwd", "docs/decisions/review.md"):
            with self.subTest(filename=filename):
                self.manifest["policies"][0]["filename"] = filename
                self.save_manifest()
                with self.assertRaisesRegex(self.loader.DocumentLoadError, "path"):
                    self.load()

    def test_policy_symlink_outside_root_is_rejected(self) -> None:
        path = self.root / self.manifest["policies"][0]["filename"]
        outside = self.base / "outside.md"  # Still inside the real project's .local/.
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "outside.*root"):
            self.load()

    def test_manifest_symlink_outside_root_is_rejected(self) -> None:
        outside = self.base / "outside.json"
        outside.write_bytes(self.manifest_path.read_bytes())
        self.manifest_path.unlink()
        self.manifest_path.symlink_to(outside)
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "outside.*root"):
            self.load()

    def test_unlisted_files_are_not_loaded(self) -> None:
        for relative in ("data/policies/UNLISTED.md", "docs/decisions/review.md",
                         "evaluations/answer_key.md", "README.md", "AGENTS.md", ".local/log.md"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\xffUnrelated content")
        self.assertEqual(len(self.load()), 6)

    def test_invalid_utf8_is_rejected_after_hash_verification(self) -> None:
        self.change_source(0, b"\xffInvalid UTF-8")
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "HR-001.*UTF-8"):
            self.load()

    def test_header_metadata_must_agree_with_manifest(self) -> None:
        self.manifest["policies"][0]["department"] = "Different department"
        self.save_manifest()
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "HR-001.*Department"):
            self.load()

    def test_missing_or_duplicate_source_metadata_and_wrong_title_are_rejected(self) -> None:
        path = self.root / self.manifest["policies"][0]["filename"]
        raw = path.read_bytes()
        cases = [
            raw.replace(b"**Policy Owner:** Director of Human Resources  \n", b""),
            raw.replace(b"**Version:** 1.0", b"**Version:** 1.0\n**Version:** 1.0"),
            raw.replace(b"Offboarding Policy", b"Offboarding Rules", 1),
            b"",
        ]
        for content in cases:
            with self.subTest(content=content[:40]):
                self.change_source(0, content)
                with self.assertRaises(self.loader.DocumentLoadError):
                    self.load()

    def test_draft_header_date_cannot_be_used_as_effective_date(self) -> None:
        path = self.root / self.manifest["policies"][1]["filename"]
        raw = path.read_bytes().replace(
            "Not effective — pending review and approval".encode(), b"2026-10-20"
        )
        self.change_source(1, raw)
        with self.assertRaisesRegex(self.loader.DocumentLoadError, "Effective Date"):
            self.load()

    def test_unsupported_mode_and_non_date_reference_are_rejected(self) -> None:
        for mode in ("all", "approved", ""):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                self.loader.DocumentLoadError, "mode"
            ):
                self.load(mode)
        for value in ("2026-10-22", datetime(2026, 10, 22)):
            with self.subTest(value=value), self.assertRaisesRegex(
                self.loader.DocumentLoadError, "reference_date"
            ):
                self.load(reference_date=value)

    def test_cli_displays_summary_and_modes_without_policy_body(self) -> None:
        for mode, count in (("draft_review", 6), ("approved_only", 1)):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "northstar_atlas.ingestion",
                     "--project-root", str(self.root), "--mode", mode,
                     "--reference-date", "2026-10-22"],
                    cwd=self.root, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Loaded {count} document(s)", result.stdout)
                self.assertEqual(result.stdout.count("PASS"), count)
                self.assertIn("Employee Role Change & Offboarding Policy", result.stdout)
                self.assertNotIn("This policy establishes procedures", result.stdout)
                self.assertEqual(result.stderr, "")

    def test_cli_errors_are_nonzero_without_dumping_bad_content(self) -> None:
        path = self.root / self.manifest["policies"][0]["filename"]
        path.write_bytes(b"SYNTHETIC_DO_NOT_PRINT")
        result = subprocess.run(
            [sys.executable, "-B", "-m", "northstar_atlas.ingestion",
             "--project-root", str(self.root), "--mode", "draft_review"],
            cwd=self.root, text=True, capture_output=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("hash mismatch", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("SYNTHETIC_DO_NOT_PRINT", result.stderr + result.stdout)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
