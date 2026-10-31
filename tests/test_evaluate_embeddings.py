import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.evaluation import evaluate_embeddings as evaluator


class EmbeddingEvaluationTests(unittest.TestCase):
    def test_normalize_rows_rejects_zero_vectors(self):
        with self.assertRaisesRegex(ValueError, "zero-length"):
            evaluator.normalize_rows(np.array([[0.0, 0.0]], dtype=np.float32))

    def test_load_saved_embeddings_validates_shape_and_finiteness(self):
        chunks = [{"chunk_id": "chunk_0", "source": "doc.txt", "text": "text"}]
        with tempfile.TemporaryDirectory() as directory:
            embedding_path = Path(directory) / "embeddings.npy"
            metadata_path = Path(directory) / "metadata.json"
            embedding_path.write_bytes(b"")
            metadata_path.write_text(json.dumps(chunks), encoding="utf-8")
            with patch.object(evaluator.embedding_generator, "EMBEDDINGS_PATH", embedding_path), patch.object(
                evaluator.embedding_generator, "METADATA_PATH", metadata_path
            ), patch("numpy.load", return_value=np.zeros((1, evaluator.EXPECTED_DIMENSIONS), dtype=np.float32)):
                with self.assertRaisesRegex(ValueError, "zero-length"):
                    evaluator.load_saved_embeddings(chunks)

    def test_missing_artifacts_use_existing_generator(self):
        with patch.object(evaluator.embedding_generator, "EMBEDDINGS_PATH", Path("missing.npy")), patch.object(
            evaluator.embedding_generator, "METADATA_PATH", Path("missing.json")
        ), patch.object(evaluator.embedding_generator, "main") as generate:
            self.assertTrue(evaluator.ensure_embedding_artifacts())
        generate.assert_called_once_with()

    def test_semantic_separation_prefers_relevant_chunk(self):
        class FakeModel:
            def encode(self, value, **kwargs):
                if isinstance(value, str):
                    return np.array([1.0, 0.0], dtype=np.float32)
                return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        chunks = [
            {"chunk_id": "relevant", "text": "related", "source": "doc.txt"},
            {"chunk_id": "other", "text": "unrelated", "source": "doc.txt"},
        ]
        questions = [{"question": "related?", "relevant_chunk_ids": ["relevant"]}]
        with patch.object(evaluator, "EXPECTED_DIMENSIONS", 2):
            result = evaluator.evaluate_semantic_separation(FakeModel(), questions, chunks)
        self.assertEqual(result["relevant_vs_unrelated_pairwise_accuracy"], 1.0)
        self.assertGreater(
            result["mean_relevant_similarity"], result["mean_unrelated_similarity"]
        )

    def test_retrieval_pipeline_reports_hits(self):
        class FakeIndex:
            ntotal = 2
            d = 384

            def search(self, query, top_k):
                return None, np.array([[0, 1]])

        fake_model = object()
        metadata = [
            {"chunk_id": "relevant", "vector_id": 0},
            {"chunk_id": "other", "vector_id": 1},
        ]
        questions = [{"question": "related?", "relevant_chunk_ids": ["relevant"]}]
        with patch.object(evaluator.retrieve, "load_index_and_metadata", return_value=(FakeIndex(), metadata)), patch.object(
            evaluator.retrieve, "embed_question", return_value=np.zeros((1, 384), dtype=np.float32)
        ):
            result = evaluator.evaluate_retrieval_pipeline(fake_model, questions)
        self.assertEqual(result["hit_rate_at_1"], 1.0)
        self.assertEqual(result["hit_rate_at_k"], 1.0)


if __name__ == "__main__":
    unittest.main()
