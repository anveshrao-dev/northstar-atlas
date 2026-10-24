"""Adapter contract tests are offline; real-model tests require an explicit flag."""

from datetime import date
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from northstar_atlas import local_embeddings as local
from northstar_atlas.embeddings import EmbeddingError, ProviderError, embed_chunks, embed_query
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    model_max_length = 256

    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [[1] * (len(text.split()) + 2) for text in texts]}


class FakeArray(list):
    def tolist(self):
        return list(self)


class FakeEncoder:
    max_seq_length = 256
    default_prompt_name = None

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return FakeArray([[1.0] + [0.0] * 383 for _ in texts])


class LocalAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / ".local/tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "repo"
        self.root.mkdir()
        (self.root / "pyproject.toml").write_text('[project]\nname = "northstar-atlas"\n')
        self.snapshot = local.model_snapshot(self.root)
        for name in local.MODEL_FILES:
            path = self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        env_patch = patch.dict(os.environ)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        self.encoder = FakeEncoder()
        loader_patch = patch.object(local, "_load_encoder", return_value=self.encoder)
        self.loader = loader_patch.start()
        self.addCleanup(loader_patch.stop)

    def test_loads_only_pinned_project_local_snapshot_and_reports_model(self):
        provider = local.MiniLMProvider(self.root)
        self.loader.assert_called_once()
        self.assertEqual(self.loader.call_args.args[0], self.snapshot)
        self.assertEqual(provider.model.model_id, local.MODEL_ID)
        self.assertEqual(provider.model.revision, local.MODEL_REVISION)
        self.assertEqual(provider.model.dimension, 384)
        self.assertEqual(provider.model.normalization, "l2")
        self.assertEqual(provider.max_input_tokens, 256)
        for name in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
                     "SENTENCE_TRANSFORMERS_HOME"):
            self.assertTrue(Path(os.environ[name]).is_relative_to(self.root / ".local"))
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

    def test_token_counts_include_special_tokens_without_truncation(self):
        provider = local.MiniLMProvider(self.root)
        self.assertEqual(provider.token_counts([" one two ", "three"]), [4, 3])
        texts, kwargs = self.encoder.tokenizer.calls[-1]
        self.assertEqual(texts, ["one two", "three"])
        self.assertFalse(kwargs["truncation"])
        self.assertTrue(kwargs["add_special_tokens"])
        self.assertFalse(kwargs["padding"])

    def test_exact_limit_is_accepted_and_encode_options_are_explicit(self):
        provider = local.MiniLMProvider(self.root)
        vector = provider.embed_query("word " * 254)
        self.assertEqual(len(vector), 384)
        _, kwargs = self.encoder.calls[-1]
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertTrue(kwargs["convert_to_numpy"])
        self.assertFalse(kwargs["show_progress_bar"])
        self.assertEqual(kwargs["precision"], "float32")
        self.assertEqual(kwargs["prompt"], "")
        self.assertEqual(kwargs["device"], "cpu")

    def test_over_limit_rejects_whole_batch_before_encode_without_echoing_text(self):
        provider = local.MiniLMProvider(self.root)
        long_text = "PRIVATE " * 255
        with self.assertRaises(local.InputTooLongError) as caught:
            provider.embed_documents(["short", long_text])
        self.assertIn("position 1=257", str(caught.exception))
        self.assertIn("256", str(caught.exception))
        self.assertNotIn("PRIVATE", str(caught.exception))
        self.assertEqual(self.encoder.calls, [])
        with self.assertRaises(local.InputTooLongError):
            provider.embed_query(long_text)
        self.assertEqual(self.encoder.calls, [])

    def test_blank_input_is_rejected_and_empty_batch_does_no_encoding(self):
        provider = local.MiniLMProvider(self.root)
        for texts in ([""], [" \n"], [None], "not a batch"):
            with self.subTest(texts=texts), self.assertRaises(EmbeddingError):
                provider.embed_documents(texts)
        self.assertEqual(provider.embed_documents([]), [])
        self.assertEqual(self.encoder.calls, [])

    def test_missing_model_does_not_fall_back_to_download(self):
        (self.snapshot / "model.safetensors").unlink()
        with self.assertRaisesRegex(EmbeddingError, "download"):
            local.MiniLMProvider(self.root)
        self.loader.assert_not_called()

    def test_cache_symlink_outside_supplied_project_is_rejected(self):
        outside = self.root.parent / "outside-cache"
        outside.mkdir()
        (self.root / ".local/torch").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(EmbeddingError, "outside"):
            local.MiniLMProvider(self.root)
        self.loader.assert_not_called()

    def test_model_file_symlink_outside_supplied_project_is_rejected(self):
        path = self.snapshot / "model.safetensors"
        path.unlink()
        outside = self.root.parent / "outside-model"
        outside.write_text("fixture")
        path.symlink_to(outside)
        with self.assertRaisesRegex(EmbeddingError, "outside"):
            local.MiniLMProvider(self.root)
        self.loader.assert_not_called()

    def test_model_load_error_is_reported_without_dependency_changes(self):
        self.loader.side_effect = RuntimeError("MODEL_LOAD_FAILURE")
        with self.assertRaises(ProviderError):
            local.MiniLMProvider(self.root)
        self.assertEqual(self.loader.call_count, 1)

    def test_unexpected_model_dimension_or_prompt_is_rejected(self):
        self.encoder.get_sentence_embedding_dimension = lambda: 768
        with self.assertRaisesRegex(EmbeddingError, "configuration"):
            local.MiniLMProvider(self.root)
        self.encoder.get_sentence_embedding_dimension = lambda: 384
        self.encoder.default_prompt_name = "unexpected"
        with self.assertRaisesRegex(EmbeddingError, "configuration"):
            local.MiniLMProvider(self.root)

    def test_wrong_project_root_is_rejected_before_loading(self):
        (self.root / "pyproject.toml").write_text('[project]\nname = "different-project"\n')
        with self.assertRaises(EmbeddingError):
            local.MiniLMProvider(self.root)
        self.loader.assert_not_called()

    def test_download_is_explicit_pinned_and_limited_to_required_files(self):
        download = Mock(return_value=str(self.snapshot))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            self.assertEqual(local.download_model(self.root), self.snapshot)
        options = download.call_args.kwargs
        self.assertEqual(options["repo_id"], local.MODEL_ID)
        self.assertEqual(options["revision"], local.MODEL_REVISION)
        self.assertEqual(options["allow_patterns"], list(local.MODEL_FILES))
        self.assertFalse(options["token"])
        self.assertTrue(Path(options["cache_dir"]).is_relative_to(self.root / ".local"))

    def test_download_rejects_outside_blob_directory_before_network_access(self):
        outside = self.root.parent / "outside-blobs"
        outside.mkdir()
        (self.snapshot.parents[1] / "blobs").symlink_to(outside, target_is_directory=True)
        download = Mock(return_value=str(self.snapshot))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaisesRegex(EmbeddingError, "outside"):
                local.download_model(self.root)
        download.assert_not_called()

    def test_cli_invalid_arguments_and_missing_model_fail_without_traceback(self):
        for args in (["inspect", "--samples", "0"],
                     ["inspect", "--reference-date", "20261022"]):
            result = subprocess.run(
                [sys.executable, "-B", "-m", "northstar_atlas.local_embeddings", *args],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Traceback", result.stderr)
        (self.snapshot / "model.safetensors").unlink()
        result = subprocess.run(
            [sys.executable, "-B", "-m", "northstar_atlas.local_embeddings", "inspect",
             "--project-root", str(self.root)], cwd=ROOT, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


@unittest.skipUnless(os.environ.get("NORTHSTAR_RUN_MODEL_TESTS") == "1",
                     "Set NORTHSTAR_RUN_MODEL_TESTS=1 for cached real-model checks")
class RealMiniLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = local.MiniLMProvider(ROOT)
        documents = load_documents(ROOT, mode="draft_review", reference_date=date(2026, 10, 22))
        cls.chunks = [c for document in documents for c in chunk_document(document).chunks]
        cls.samples = [chunk_document(document).chunks[0] for document in documents[:3]]

    def test_real_vectors_mapping_query_and_normalization(self):
        results = embed_chunks(self.samples, self.provider)
        self.assertEqual(len(results), 3)
        for chunk, result in zip(self.samples, results, strict=True):
            self.assertIs(result.chunk, chunk)
            self.assertEqual(result.chunk_id, chunk.chunk_id)
            self.assertEqual(len(result.vector), 384)
            self.assertTrue(all(math.isfinite(x) for x in result.vector))
            self.assertAlmostEqual(math.hypot(*result.vector), 1.0, places=5)
            single = embed_chunks([chunk], self.provider)[0]
            self.assertLess(max(abs(a-b) for a, b in zip(result.vector, single.vector, strict=True)), 1e-5)
        query = embed_query("Who approves a production Terraform plan?", self.provider)
        self.assertEqual(query.model, results[0].model)
        self.assertEqual(len(query.vector), 384)
        self.assertAlmostEqual(math.hypot(*query.vector), 1.0, places=5)

    def test_real_corpus_oversize_is_detected_and_rejected_before_encode(self):
        counts = self.provider.token_counts([c.text for c in self.chunks])
        oversized = [c for c, n in zip(self.chunks, counts, strict=True) if n > self.provider.max_input_tokens]
        self.assertTrue(oversized)
        self.assertTrue(any(c.document_id == "PLAT-001" and c.section_id == "6.1" for c in oversized))
        with patch.object(self.provider._encoder, "encode") as encode:
            with self.assertRaises(local.InputTooLongError):
                self.provider.embed_documents([oversized[0].text])
            encode.assert_not_called()


if __name__ == "__main__":
    unittest.main()
