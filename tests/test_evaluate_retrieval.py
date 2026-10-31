import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import evaluate_retrieval as evaluator


class RetrievalMetricTests(unittest.TestCase):
    def test_metrics_count_unique_results_and_keep_requested_k_denominator(self):
        result = evaluator.calculate_metrics(
            ["chunk_a", "chunk_b", "chunk_a"],
            {"chunk_a", "chunk_c"},
            3,
        )
        self.assertEqual(
            result,
            {
                "hit_rate": 1.0,
                "recall": 0.5,
                "precision": 1 / 3,
                "mrr": 1.0,
            },
        )

    def test_metrics_return_zero_for_empty_results(self):
        result = evaluator.calculate_metrics([], {"chunk_a"}, 5)
        self.assertEqual(
            result,
            {
                "hit_rate": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "mrr": 0.0,
            },
        )

    def test_empty_relevant_set_has_safe_zero_recall(self):
        result = evaluator.calculate_metrics(["chunk_a"], set(), 1)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["hit_rate"], 0.0)

    def test_non_positive_k_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            evaluator.calculate_metrics([], {"chunk_a"}, 0)


class RetrievalDatasetTests(unittest.TestCase):
    def test_skips_unanswerable_missing_and_ambiguous_labels(self):
        records = [
            {
                "id": "answerable",
                "question": "Question with relevance",
                "relevant_chunk_ids": ["chunk_a"],
            },
            {
                "id": "unanswerable",
                "question": "No answer",
                "question_category": "answer_absent",
                "relevant_chunk_ids": [],
            },
            {"id": "missing-label", "question": "No labels"},
            {
                "id": "ambiguous-empty",
                "question": "Empty but not marked unanswerable",
                "relevant_chunk_ids": [],
            },
            {"id": "empty-question", "question": " ", "relevant_chunk_ids": []},
        ]
        evaluated, skipped = evaluator.prepare_evaluation_records(
            records, {"chunk_a"}
        )
        self.assertEqual([item["id"] for item in evaluated], ["answerable"])
        self.assertEqual(
            {item["id"]: item["reason"] for item in skipped},
            {
                "unanswerable": "Unanswerable question has no relevant chunks.",
                "missing-label": "Missing relevant_chunk_ids label.",
                "ambiguous-empty": (
                    "Empty relevance labels; answerability is not explicit."
                ),
                "empty-question": "Question is missing or empty.",
            },
        )

    def test_unknown_relevant_chunk_is_a_dataset_error(self):
        records = [
            {
                "id": "bad-label",
                "question": "Question",
                "relevant_chunk_ids": ["not-in-index"],
            }
        ]
        with self.assertRaisesRegex(ValueError, "absent from FAISS metadata"):
            evaluator.prepare_evaluation_records(records, {"chunk_a"})

    def test_missing_dataset_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                evaluator.load_golden_questions(missing_path)


class RetrievalExecutionTests(unittest.TestCase):
    def test_faiss_position_mapping_deduplicates_and_ignores_padding(self):
        class FakeIndex:
            def search(self, query, top_k):
                return None, [[0, 1, 1, 2, -1][:top_k]]

        metadata = [
            {"chunk_id": "chunk_a"},
            {"chunk_id": "chunk_a"},
            {"chunk_id": "chunk_b"},
        ]
        result = evaluator.retrieve_chunk_ids(None, FakeIndex(), metadata, 5)
        self.assertEqual(result, ["chunk_a", "chunk_b"])

    def test_evaluate_writes_report_using_existing_retriever(self):
        class FakeIndex:
            ntotal = 5
            d = 384

            def search(self, query, top_k):
                self.requested_top_k = top_k
                return None, [[0, 1, 2, 3, 4]]

        index = FakeIndex()
        metadata = [{"chunk_id": f"chunk_{position}"} for position in range(5)]
        golden_records = [
            {
                "id": "answerable",
                "question": "Question?",
                "relevant_chunk_ids": ["chunk_2"],
            },
            {
                "id": "unanswerable",
                "question": "No answer?",
                "question_category": "answer_absent",
                "relevant_chunk_ids": [],
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            golden_path = Path(directory) / "goldens.json"
            report_path = Path(directory) / "reports" / "report.json"
            golden_path.write_text(json.dumps(golden_records), encoding="utf-8")
            with (
                patch.object(
                    evaluator, "load_index_and_metadata", return_value=(index, metadata)
                ),
                patch.object(evaluator, "SentenceTransformer", return_value=object()),
                patch.object(evaluator, "embed_question", return_value="query"),
                patch.object(evaluator.time, "perf_counter", side_effect=[1.0, 1.025]),
            ):
                report = evaluator.evaluate(golden_path, report_path)

            self.assertTrue(report_path.is_file())
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report, report)
            self.assertEqual(report["evaluated_questions"], 1)
            self.assertEqual(report["skipped_question_count"], 1)
            self.assertEqual(report["skipped_questions"][0]["id"], "unanswerable")
            self.assertEqual(index.requested_top_k, 5)
            self.assertAlmostEqual(report["average_retrieval_latency_ms"], 25.0)
            self.assertEqual(report["metrics_by_k"]["1"]["hit_rate"], 0.0)
            self.assertEqual(report["metrics_by_k"]["3"]["hit_rate"], 1.0)
            self.assertEqual(report["metrics_by_k"]["5"]["recall"], 1.0)
            self.assertEqual(
                report["retriever_configuration"]["embedding_model"],
                evaluator.MODEL_NAME,
            )


if __name__ == "__main__":
    unittest.main()
