import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.evaluation import evaluate_llama_guard
from src.evaluation import evaluate_llama_guard_e2e
from src.guardrails import llama_guard
from src.rag import rag_chat


class LlamaGuardEvaluationTests(unittest.TestCase):
    def test_metrics_include_false_positives_false_negatives_and_confusion_matrix(self):
        records = [
            {"expected_label": "UNSAFE", "actual_prediction": "UNSAFE"},
            {"expected_label": "SAFE", "actual_prediction": "UNSAFE"},
            {"expected_label": "SAFE", "actual_prediction": "SAFE"},
            {"expected_label": "UNSAFE", "actual_prediction": "SAFE"},
        ]
        metrics = evaluate_llama_guard.calculate_metrics(records)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "true_positive_unsafe": 1,
                "false_positive_unsafe": 1,
                "true_negative_safe": 1,
                "false_negative_safe": 1,
            },
        )

    def test_input_and_output_evaluations_select_the_requested_guard_type(self):
        cases = [
            {
                "id": "case-safe",
                "category": "reviewed",
                "text": "labeled text",
                "expected_label": "SAFE",
            }
        ]
        with patch.object(
            evaluate_llama_guard.llama_guard, "check_safety", return_value=(True, "safe")
        ) as check:
            input_report = evaluate_llama_guard.evaluate(cases, "input")
            output_report = evaluate_llama_guard.evaluate(cases, "output")

        self.assertEqual(input_report["guard_configuration"]["guard_type"], "input")
        self.assertEqual(output_report["guard_configuration"]["guard_type"], "output")
        self.assertEqual(
            [call.kwargs["guard_type"] for call in check.call_args_list],
            ["input", "output"],
        )

    def test_output_evaluation_uses_mocked_ollama_response(self):
        cases = [
            {
                "id": "labeled-output",
                "category": "reviewed",
                "text": "reviewed assistant response",
                "expected_label": "UNSAFE",
            }
        ]
        with patch.object(
            llama_guard, "_call_ollama", return_value="unsafe S13"
        ) as call_ollama:
            report = evaluate_llama_guard.evaluate(cases, guard_type="output")

        self.assertEqual(
            report["metrics"]["confusion_matrix"]["true_positive_unsafe"], 1
        )
        prompt = call_ollama.call_args.args[0]
        self.assertIn("Assistant response:", prompt)
        self.assertIn("reviewed assistant response", prompt)

    def test_write_reports_refuses_to_overwrite_historical_files(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "existing.json"
            markdown_path = Path(directory) / "report.md"
            json_path.write_text('{"historical": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                evaluate_llama_guard.write_reports({}, json_path, markdown_path)
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"historical": True},
            )
            self.assertFalse(markdown_path.exists())

    def test_run_guarded_rag_blocks_unsafe_input_before_retrieval(self):
        with patch.object(
            rag_chat,
            "check_safety",
            return_value=(False, "input was blocked by Llama Guard 2 as unsafe."),
        ), patch.object(rag_chat, "retrieve_chunks") as retrieve, patch.object(
            rag_chat, "generate_answer"
        ) as generate:
            result = rag_chat.run_guarded_rag("unsafe query", object(), object(), [])

        self.assertEqual(result["status"], "input_blocked")
        retrieve.assert_not_called()
        generate.assert_not_called()

    def test_run_guarded_rag_withholds_unsafe_generated_answer(self):
        chunks = [{"chunk_id": "chunk-1", "text": "context", "score": 0.9}]
        with patch.object(
            rag_chat,
            "check_safety",
            side_effect=[
                (True, "input is safe"),
                (False, "output was blocked by Llama Guard 2 as unsafe."),
            ],
        ) as check, patch.object(
            rag_chat, "retrieve_chunks", return_value=chunks
        ) as retrieve, patch.object(
            rag_chat, "generate_answer", return_value="unsafe generated answer"
        ):
            result = rag_chat.run_guarded_rag("safe query", object(), object(), [])

        self.assertEqual(result["status"], "output_blocked")
        self.assertNotIn("answer", result)
        retrieve.assert_called_once()
        self.assertEqual(
            [call.kwargs["guard_type"] for call in check.call_args_list],
            ["input", "output"],
        )

    def test_main_never_prints_a_blocked_generated_answer(self):
        blocked_result = {
            "status": "output_blocked",
            "input_safe": True,
            "input_reason": "The input is safe.",
            "retrieved_chunks": [],
            "output_safe": False,
            "output_reason": "The output was blocked by Llama Guard 2 as unsafe.",
        }
        with patch.object(
            rag_chat, "load_index_and_metadata", return_value=(object(), [])
        ), patch.object(
            rag_chat, "SentenceTransformer", return_value=object()
        ), patch.object(
            rag_chat, "run_guarded_rag", return_value=blocked_result
        ), patch(
            "builtins.input", side_effect=["question", "exit"]
        ), patch("builtins.print") as print_output:
            rag_chat.main()

        printed_text = "\n".join(
            " ".join(str(argument) for argument in call.args)
            for call in print_output.call_args_list
        )
        self.assertIn("Safety block (output)", printed_text)
        self.assertNotIn("unsafe generated answer", printed_text)

    def test_end_to_end_report_separates_input_metrics_and_pipeline_outcomes(self):
        records = [
            {
                "id": "unsafe-input",
                "category": "reviewed",
                "text": "query 1",
                "expected_label": "UNSAFE",
            },
            {
                "id": "safe-input-output-blocked",
                "category": "reviewed",
                "text": "query 2",
                "expected_label": "SAFE",
            },
        ]
        pipeline_results = [
            {
                "status": "input_blocked",
                "input_safe": False,
                "input_reason": "The input was blocked by Llama Guard 2 as unsafe.",
                "output_safe": None,
                "output_reason": None,
            },
            {
                "status": "output_blocked",
                "input_safe": True,
                "input_reason": "The input is safe.",
                "output_safe": False,
                "output_reason": "The output was blocked by Llama Guard 2 as unsafe.",
            },
        ]
        with patch.object(
            rag_chat, "run_guarded_rag", side_effect=pipeline_results
        ):
            report = evaluate_llama_guard_e2e.evaluate(
                records, object(), object(), []
            )

        self.assertEqual(
            report["input_classification_metrics"]["accuracy"], 1.0
        )
        self.assertEqual(report["pipeline_outcomes"]["input_blocked"], 1)
        self.assertEqual(report["pipeline_outcomes"]["output_blocked"], 1)
        self.assertFalse(report["pipeline_outcomes"]["output_labels_available"])
        self.assertEqual(report["pipeline_outcomes"]["output_unsafe_blocked"], 1)


if __name__ == "__main__":
    unittest.main()
