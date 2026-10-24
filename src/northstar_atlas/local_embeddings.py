"""Pinned CPU MiniLM adapter. Inference is offline; download is an explicit CLI action."""

import argparse
from collections.abc import Sequence
from datetime import date
from itertools import zip_longest
import math
import os
from pathlib import Path
import re
import resource
import sys
from time import perf_counter
import tomllib

from northstar_atlas.embeddings import (
    EmbeddingError, EmbeddingModel, ProviderError, embed_chunks, embed_query,
)
from northstar_atlas.ingestion.chunker import ChunkingError, chunk_document
from northstar_atlas.ingestion.document_loader import DocumentLoadError, load_documents


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = (
    "1_Pooling/config.json", "config.json", "config_sentence_transformers.json",
    "model.safetensors", "modules.json", "sentence_bert_config.json",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
)
CACHE_PATHS = {
    "HF_HOME": ".local/huggingface",
    "HF_HUB_CACHE": ".local/huggingface/hub",
    "HF_ASSETS_CACHE": ".local/huggingface/assets",
    "TORCH_HOME": ".local/torch",
    "SENTENCE_TRANSFORMERS_HOME": ".local/sentence-transformers",
    "XDG_CACHE_HOME": ".local/cache",
}


class InputTooLongError(EmbeddingError):
    """An input exceeds the model's limit; no encoding has been attempted."""


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise EmbeddingError("model or cache path resolves outside the project root")
    return path


def _root(project_root: Path) -> Path:
    try:
        root = Path(project_root).resolve(strict=True)
        config = tomllib.loads(_inside(root, "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError):
        raise EmbeddingError("project root must contain a readable Northstar Atlas pyproject.toml") from None
    if config.get("project", {}).get("name") != "northstar-atlas":
        raise EmbeddingError("project root is not a northstar-atlas project")
    return root


def model_snapshot(project_root: Path) -> Path:
    root = _root(project_root)
    return _inside(root, f".local/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/{MODEL_REVISION}")


def _configure_cache(root: Path, *, offline: bool) -> None:
    # Validate all destinations before changing this process's environment.
    paths = {name: _inside(root, relative) for name, relative in CACHE_PATHS.items()}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"  # Use HTTP downloads; avoid an additional Xet cache.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _verify_snapshot(root: Path, snapshot: Path) -> None:
    for name in MODEL_FILES:
        path = _inside(root, str((snapshot / name).relative_to(root)))
        if not path.is_file():
            raise EmbeddingError("pinned MiniLM files are missing; run the explicit download command first")


def download_model(project_root: Path) -> Path:
    """Fetch only the approved revision/files. Run separately from offline inference."""
    root = _root(project_root)
    snapshot = model_snapshot(root)
    # Check pre-existing paths too: a hub snapshot may contain symlinks to blobs.
    for name in MODEL_FILES:
        _inside(root, str((snapshot / name).relative_to(root)))
    hub = _inside(root, CACHE_PATHS["HF_HUB_CACHE"])
    # Hub downloads also write blobs and locks outside the snapshot directory.
    # rglob does not recurse into symlink directories; validate the links themselves.
    for entry in hub.rglob("*"):
        if entry.is_symlink():
            _inside(root, str(entry.relative_to(root)))
    _configure_cache(root, offline=False)
    try:
        from huggingface_hub import snapshot_download

        downloaded = Path(snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, allow_patterns=list(MODEL_FILES),
            cache_dir=str(root / CACHE_PATHS["HF_HUB_CACHE"]), token=False, max_workers=2,
        ))
    except Exception as exc:
        raise ProviderError(f"model download failed ({type(exc).__name__}); no dependency changes attempted") from None
    if downloaded.resolve() != snapshot.resolve():
        raise EmbeddingError("download did not return the pinned project-local snapshot")
    _verify_snapshot(root, snapshot)
    return snapshot


def _load_encoder(snapshot: Path, cache: Path):
    # Lazy imports keep contract tests independent of these optional runtime imports.
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    return SentenceTransformer(
        str(snapshot), device="cpu", local_files_only=True, trust_remote_code=False,
        token=False, cache_folder=str(cache), model_kwargs={"use_safetensors": True},
    )


class MiniLMProvider:
    """One pinned model, CPU float32, mean pooling from its config, L2 output."""

    def __init__(self, project_root: Path):
        root = _root(project_root)
        snapshot = model_snapshot(root)
        _verify_snapshot(root, snapshot)
        _configure_cache(root, offline=True)
        started = perf_counter()
        try:
            self._encoder = _load_encoder(snapshot, root / CACHE_PATHS["SENTENCE_TRANSFORMERS_HOME"])
        except Exception as exc:
            raise ProviderError(f"local model loading failed ({type(exc).__name__}); no upgrade or download attempted") from None
        self.load_seconds = perf_counter() - started
        if (self._encoder.get_sentence_embedding_dimension() != 384
                or self._encoder.max_seq_length != 256
                or self._encoder.tokenizer.model_max_length != 256
                or self._encoder.default_prompt_name is not None):
            raise EmbeddingError("unexpected MiniLM configuration; expected 384 dimensions, 256 tokens, no prompt")
        self.model = EmbeddingModel(MODEL_ID, MODEL_REVISION, 384, "l2")

    @property
    def max_input_tokens(self) -> int:
        return 256

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        """Count exact WordPiece inputs with special tokens, with no truncation.

        The pinned SentenceTransformer preprocessing strips surrounding whitespace;
        its tokenizer performs the model's own case handling. No prompt is added.
        """
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes, bytearray)):
            raise EmbeddingError("texts must be a sequence of nonempty strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("each input must be a nonempty string")
        if not texts:
            return []
        encoded = self._encoder.tokenizer(
            [text.strip() for text in texts], add_special_tokens=True,
            truncation=False, padding=False, return_attention_mask=False,
            return_token_type_ids=False, verbose=False,
        )
        # verbose=False suppresses the tokenizer's generic overlength warning;
        # check_inputs and the CLI report exact overlength counts explicitly.
        return [len(ids) for ids in encoded["input_ids"]]

    def check_inputs(self, texts: Sequence[str]) -> list[int]:
        counts = self.token_counts(texts)
        oversized = [(index, count) for index, count in enumerate(counts) if count > self.max_input_tokens]
        if oversized:
            detail = ", ".join(f"position {index}={count}" for index, count in oversized)
            raise InputTooLongError(f"inputs exceed {self.max_input_tokens} tokens: {detail}; nothing encoded")
        return counts

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.check_inputs(texts)  # Preflight the entire batch before encode can truncate.
        if not texts:
            return []
        return self._encoder.encode(
            list(texts), batch_size=8, device="cpu", precision="float32",
            show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True, prompt="",
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def _reference_date(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise argparse.ArgumentTypeError("reference date must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("reference date must be a valid calendar date") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pinned MiniLM explicitly or inspect local sample embeddings.")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download approved model files; requires network access.")
    inspect = commands.add_parser("inspect", help="Run offline sample encoding without saving or printing vectors.")
    for command in (download, inspect):
        command.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    inspect.add_argument("--mode", choices=("approved_only", "draft_review"), default="approved_only")
    inspect.add_argument("--reference-date", type=_reference_date, help="Default: today's UTC date.")
    inspect.add_argument("--samples", type=int, default=3, help="Select up to this many chunks, round-robin across documents.")
    inspect.add_argument("--query", default="Who approves a production Terraform plan?")
    args = parser.parse_args()
    if args.command == "inspect" and args.samples <= 0:
        parser.error("--samples must be a positive integer")
    try:
        if args.command == "download":
            started = perf_counter()
            path = download_model(args.project_root)
            print(f"Model: {MODEL_ID}; revision: {MODEL_REVISION}")
            print(f"Required model bytes: {sum((path / name).stat().st_size for name in MODEL_FILES)}")
            print(f"Download/cache check seconds: {perf_counter() - started:.3f}")
            return 0
        documents = load_documents(args.project_root, mode=args.mode, reference_date=args.reference_date)
        results = [chunk_document(document) for document in documents]
        chunks = [chunk for result in results for chunk in result.chunks]
        if not chunks:
            raise EmbeddingError("no policy chunks are available in the requested mode/date")
        provider = MiniLMProvider(args.project_root)
        counts = provider.token_counts([chunk.text for chunk in chunks])
        count_by_id = {chunk.chunk_id: count for chunk, count in zip(chunks, counts, strict=True)}
        oversized = [(chunk, count) for chunk, count in zip(chunks, counts, strict=True)
                     if count > provider.max_input_tokens]
        print(f"Model: {provider.model.model_id}; revision: {provider.model.revision}")
        print(f"Dimension: {provider.model.dimension}; normalization: l2; device: cpu; torch threads: 2")
        print(f"Documents: {len(documents)}; chunks inspected: {len(chunks)}; mode: {args.mode}")
        print(f"Token limit: 256 (including special tokens); token range: {min(counts)}–{max(counts)}")
        print(f"Over-limit corpus chunks: {len(oversized)}/{len(chunks)}")
        for chunk, count in oversized:
            print(f"WARNING: {chunk.citation}: {count} tokens > 256", file=sys.stderr)
        # Selection never skips over-limit chunks to manufacture a successful run.
        round_robin = [c for row in zip_longest(*(r.chunks for r in results)) for c in row if c is not None]
        selected = round_robin[:args.samples]
        provider.check_inputs([chunk.text for chunk in selected])
        query_tokens = provider.check_inputs([args.query])[0]
        started = perf_counter()
        embedded = embed_chunks(selected, provider)
        document_seconds = perf_counter() - started
        started = perf_counter()
        query = embed_query(args.query, provider)
        query_seconds = perf_counter() - started
        for item in embedded:
            print(f"Sample {item.chunk.citation}: {count_by_id[item.chunk_id]} tokens; "
                  f"{len(item.vector)} dimensions; norm={math.hypot(*item.vector):.6f}; chunk_id={item.chunk_id}")
        print(f"Encoded chunks: {len(embedded)}; queries: 1; query tokens: {query_tokens}; query norm={math.hypot(*query.vector):.6f}")
        print(f"Provider initialization seconds (imports + model load): {provider.load_seconds:.3f}")
        print(f"Document embedding seconds: {document_seconds:.3f}; query embedding seconds: {query_seconds:.3f}")
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        print(f"Peak process RSS: {peak_bytes / 1024**2:.1f} MiB")
        return 0
    except (EmbeddingError, DocumentLoadError, ChunkingError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
