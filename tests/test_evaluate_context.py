import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import evaluate_context as evaluator


class ContextAssemblyTests(unittest.TestCase):
    def test_existing_generator_preserves_chunk_text_and_question(self):
        chunks = [
            {"chunk_id": "chunk_a", "text": "Alpha text", "score": 0.9},
            {"chunk_id": "chunk_b", "text": "Beta text", "score": 0.8},
        ]
        captured = evaluator.inspect_generator_payload("What?", chunks)
        content = captured["user_content"]
        self.assertIn("[chunk_a]\nAlpha text", content)
        self.assertIn("[chunk_b]\nBeta text", content)
        self.assertIn("\n\n[chunk_b]", content)
        self.assertIn("Question: What?", content)

    def test_existing_generator_does_not_drop_duplicate_or_empty_entries(self):
        chunks = [
            {"chunk_id": "chunk_a", "text": "Alpha", "score": 0.9},
            {"chunk_id": "chunk_a", "text": "Alpha", "score": 0.8},
            {"chunk_id": "chunk_empty", "text": "", "score": 0.1},
        ]
        captured = evaluator.inspect_generator_payload("Question", chunks)
        content = captured["user_content"]
        self.assertEqual(content.count("[chunk_a]\nAlpha"), 2)
        self.assertIn("[chunk_empty]\n", content)

    def test_evaluate_writes_report_without_calling_llm(self):
        records = [
            {
                "id": "question_1",
                "question": "Question?",
                "relevant_chunk_ids": ["chunk_a"],
            }
        ]
        metadata = [{"chunk_id": "chunk_a"}]
        chunks = [{"chunk_id": "chunk_a", "text": "Alpha", "score": 1.0}]
        class FakeIndex:
            ntotal = 1
            d = 384

        with tempfile.TemporaryDirectory() as directory:
            golden_path = Path(directory) / "goldens.json"
            report_path = Path(directory) / "report.json"
            golden_path.write_text(json.dumps(records), encoding="utf-8")
            with (
                patch.object(evaluator, "load_answerable_goldens", return_value=records),
                patch.object(evaluator.retrieve, "load_index_and_metadata", return_value=(FakeIndex(), metadata)),
                patch.object(evaluator, "SentenceTransformer", return_value=object()),
                patch.object(evaluator.rag_chat, "retrieve_chunks", return_value=chunks),
            ):
                report = evaluator.evaluate(report_path)
            self.assertTrue(report_path.is_file())

        self.assertEqual(report["evaluated_questions"], 1)
        self.assertTrue(report["checks"]["all_retrieved_chunks_reach_generator"])


if __name__ == "__main__":
    unittest.main()
