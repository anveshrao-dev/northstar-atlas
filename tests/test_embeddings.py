"""Offline embedding contracts; the fake provider has no semantic capability."""

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from datetime import date
import hashlib
import math
from pathlib import Path
import traceback
import unittest

from northstar_atlas.embeddings import (
    EmbeddingError, EmbeddingModel, ProviderError, embed_chunks, embed_query, validate_vector,
)
from northstar_atlas.ingestion.chunker import chunk_document
from northstar_atlas.ingestion.document_loader import load_documents


ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    """Stable hash-derived vectors for contract tests only, never for retrieval."""

    def __init__(self) -> None:
        self.model = EmbeddingModel("test/fake-sha256", "test-v1", 8, "l2")
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [(digest[i] + 1) / 256 for i in range(self.model.dimension)]
        norm = math.hypot(*values)
        return [value / norm for value in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls.append(tuple(texts))
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        documents = load_documents(ROOT, mode="draft_review", reference_date=date(2026, 10, 22))
        cls.chunks = tuple(c for d in documents for c in chunk_document(d).chunks)

    def setUp(self) -> None:
        self.provider = FakeProvider()

    def test_one_vector_per_corpus_chunk_in_source_order(self) -> None:
        results = embed_chunks(self.chunks, self.provider)
        self.assertEqual(len(results), 151)
        self.assertEqual([r.chunk.chunk_id for r in results], [c.chunk_id for c in self.chunks])
        self.assertEqual(self.provider.document_calls, [tuple(c.text for c in self.chunks)])
        for result in results:
            self.assertEqual(result.vector, tuple(self.provider.vector(result.chunk.text)))
            self.assertEqual(len(result.vector), 8)

    def test_mapping_is_stable_across_repeated_calls_and_reordered_input(self) -> None:
        samples = self.chunks[:4]
        first = embed_chunks(samples, self.provider)
        self.assertEqual(first, embed_chunks(samples, FakeProvider()))
        reordered = embed_chunks(tuple(reversed(samples)), self.provider)
        self.assertEqual({r.chunk.chunk_id: r.vector for r in first},
                         {r.chunk.chunk_id: r.vector for r in reordered})
        self.assertEqual([r.chunk.chunk_id for r in reordered], [c.chunk_id for c in reversed(samples)])

    def test_all_source_metadata_and_original_chunk_objects_are_preserved(self) -> None:
        for chunk, result in zip(self.chunks, embed_chunks(self.chunks, self.provider), strict=True):
            self.assertIs(result.chunk, chunk)
            self.assertEqual(result.model, self.provider.model)
            self.assertEqual(result.chunk.citation, chunk.citation)
        with self.assertRaises(FrozenInstanceError):
            result.vector = ()

    def test_query_uses_the_query_method_and_same_model_space(self) -> None:
        query = "Who approves a production Terraform plan?"
        result = embed_query(query, self.provider)
        self.assertEqual(result.model, self.provider.model)
        self.assertEqual(result.vector, tuple(self.provider.vector(query)))
        self.assertEqual(self.provider.query_calls, [query])
        self.assertEqual(self.provider.document_calls, [])
        self.assertAlmostEqual(math.hypot(*result.vector), 1.0)

    def test_valid_numeric_vectors_are_copied_to_immutable_floats(self) -> None:
        original = [1, -2.5, 3]
        result = validate_vector(original, dimension=3)
        self.assertEqual(result, (1.0, -2.5, 3.0))
        self.assertTrue(all(type(value) is float for value in result))
        original[0] = 99
        self.assertEqual(result[0], 1.0)

    def test_wrong_dimensions_and_invalid_numeric_values_are_rejected(self) -> None:
        cases = [[], [1, 2], [1, 2, 3, 4], [True, 2, 3], [1, "2", 3],
                 [1, None, 3], [1, complex(2, 1), 3], [1, float("nan"), 3],
                 [1, float("inf"), 3], [1, float("-inf"), 3],
                 [10**1000, 2, 3], [0, 0, 0], [1.7e308, 1.7e308, 1.7e308],
                 "123", None, {0: 1, 1: 2, 2: 3}, [[1], [2], [3]]]
        for vector in cases:
            with self.subTest(vector_type=type(vector).__name__), self.assertRaises(EmbeddingError):
                validate_vector(vector, dimension=3)

    def test_invalid_dimension_is_rejected(self) -> None:
        for dimension in (0, -1, True, 3.5, "3", None):
            with self.subTest(dimension=dimension), self.assertRaises(EmbeddingError):
                validate_vector([1, 2, 3], dimension=dimension)

    def test_too_few_or_too_many_provider_vectors_are_rejected(self) -> None:
        for count in (0, 1, 3):
            class WrongCount(FakeProvider):
                def embed_documents(self, texts):
                    return [self.vector("test")] * count
            with self.subTest(count=count), self.assertRaisesRegex(EmbeddingError, "count"):
                embed_chunks(self.chunks[:2], WrongCount())

    def test_invalid_batch_shapes_are_rejected(self) -> None:
        for output in (None, "bad", {"vector": [1, 2]}, [1, 2]):
            class WrongShape(FakeProvider):
                def embed_documents(self, texts):
                    return output
            with self.subTest(output_type=type(output).__name__), self.assertRaises(EmbeddingError):
                embed_chunks(self.chunks[:2], WrongShape())

    def test_invalid_later_vector_fails_entire_call(self) -> None:
        class BadSecond(FakeProvider):
            def embed_documents(self, texts):
                return [self.vector(texts[0]), [float("nan")] * self.model.dimension]
        with self.assertRaisesRegex(EmbeddingError, "vector 1"):
            embed_chunks(self.chunks[:2], BadSecond())

    def test_query_vector_has_the_same_validation(self) -> None:
        for vector in ([1, 2], [float("nan")] * 8, [0] * 8):
            class BadQuery(FakeProvider):
                def embed_query(self, text):
                    return vector
            with self.subTest(vector=vector), self.assertRaises(EmbeddingError):
                embed_query("Question", BadQuery())

    def test_normalization_claim_is_verified_without_silently_changing_vector(self) -> None:
        class Unnormalized(FakeProvider):
            def embed_query(self, text):
                return [2.0] * 8
        with self.assertRaisesRegex(EmbeddingError, "normaliz"):
            embed_query("Question", Unnormalized())
        provider = Unnormalized()
        provider.model = replace(provider.model, normalization="none")
        self.assertEqual(embed_query("Question", provider).vector, (2.0,) * 8)

    def test_provider_errors_are_clear_without_exposing_provider_message_or_input(self) -> None:
        class Failing(FakeProvider):
            def embed_documents(self, texts):
                raise RuntimeError("PRIVATE_PROVIDER_RESPONSE " + texts[0])

            def embed_query(self, text):
                raise RuntimeError("PRIVATE_PROVIDER_RESPONSE " + text)
        for operation in (lambda: embed_chunks(self.chunks[:2], Failing()),
                          lambda: embed_query("PRIVATE_QUERY", Failing())):
            with self.assertRaises(ProviderError) as caught:
                operation()
            self.assertIn("provider failed", str(caught.exception))
            self.assertIn("RuntimeError", str(caught.exception))
            rendered = "".join(traceback.format_exception(caught.exception))
            self.assertNotIn("PRIVATE_PROVIDER_RESPONSE", rendered)
            self.assertNotIn("PRIVATE_QUERY", rendered)

    def test_empty_document_batch_does_not_call_provider(self) -> None:
        self.assertEqual(embed_chunks([], self.provider), [])
        self.assertEqual(self.provider.document_calls, [])

    def test_duplicate_ids_and_blank_chunk_text_fail_before_provider_call(self) -> None:
        for chunks in ((self.chunks[0], self.chunks[0]),
                       (replace(self.chunks[0], text=" \n"),), ("not a chunk",)):
            with self.subTest(chunks=type(chunks).__name__), self.assertRaises(EmbeddingError):
                embed_chunks(chunks, self.provider)
        self.assertEqual(self.provider.document_calls, [])

    def test_blank_or_nonstring_queries_fail_before_provider_call(self) -> None:
        for query in ("", " \n\t", None, ["question"], 123):
            with self.subTest(query=query), self.assertRaises(EmbeddingError):
                embed_query(query, self.provider)
        self.assertEqual(self.provider.query_calls, [])

    def test_model_metadata_is_explicit_validated_and_immutable(self) -> None:
        for overrides in ({"model_id": ""}, {"revision": " "}, {"revision": 123},
                          {"dimension": 0}, {"dimension": True}, {"normalization": "unknown"}):
            with self.subTest(overrides=overrides), self.assertRaises(EmbeddingError):
                replace(self.provider.model, **overrides)
        with self.assertRaises(FrozenInstanceError):
            self.provider.model.dimension = 384

    def test_unavailable_model_revision_remains_explicitly_unknown(self) -> None:
        self.provider.model = replace(self.provider.model, revision=None)
        self.assertIsNone(embed_query("Question", self.provider).model.revision)

    def test_changing_provider_model_during_a_call_is_rejected(self) -> None:
        class ChangesModel(FakeProvider):
            def embed_documents(self, texts):
                vectors = super().embed_documents(texts)
                self.model = replace(self.model, revision="different-revision")
                return vectors

            def embed_query(self, text):
                vector = super().embed_query(text)
                self.model = replace(self.model, revision="different-revision")
                return vector
        for operation in (lambda: embed_chunks(self.chunks[:2], ChangesModel()),
                          lambda: embed_query("Question", ChangesModel())):
            with self.assertRaisesRegex(EmbeddingError, "model changed"):
                operation()


if __name__ == "__main__":
    unittest.main()
