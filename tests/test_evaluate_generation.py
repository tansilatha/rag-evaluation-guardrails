import json
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from src.evaluation import evaluate_generation as evaluator


class GenerationDatasetTests(unittest.TestCase):
    def test_prepares_answerable_and_unanswerable_records(self):
        records = [
            {
                "id": "answerable",
                "question": "What is supervised learning?",
                "expected_answer": "It learns from labeled data.",
                "expected_behavior": "Answer using the trusted context.",
                "trusted_context": [{"text": "It learns from labeled data."}],
            },
            {
                "id": "unanswerable",
                "question": "What was the exact accuracy?",
                "question_category": "answer_absent",
                "expected_answer": None,
                "expected_behavior": "State that the available context does not contain enough information.",
                "trusted_context": [{"text": "The document omits accuracy."}],
            },
            {"id": "missing-answer", "question": "Question", "expected_behavior": "Answer."},
        ]
        prepared, skipped = evaluator.prepare_generation_records(records)
        self.assertEqual([item["id"] for item in prepared], ["answerable", "unanswerable"])
        self.assertEqual(skipped[0]["id"], "missing-answer")

    def test_invalid_dataset_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "goldens.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON array"):
                evaluator.load_generation_goldens(path)


class JudgeValidationTests(unittest.TestCase):
    def setUp(self):
        self.provider_patcher = patch.object(evaluator, "JUDGE_PROVIDER", "openrouter")
        self.provider_patcher.start()
        self.addCleanup(self.provider_patcher.stop)

    def test_judge_model_reads_environment_override(self):
        with patch("dotenv.load_dotenv"), patch.dict(
            "os.environ", {"JUDGE_MODEL": "test/judge-model"}
        ):
            importlib.reload(evaluator)
            self.assertEqual(evaluator.OPENROUTER_MODEL, "test/judge-model")
        importlib.reload(evaluator)

    def valid_judge(self, **overrides):
        result = {
            "correctness_score": 0.9,
            "faithfulness_score": 0.8,
            "relevance_score": 1.0,
            "explanation": "The answer is supported and addresses the question.",
            "unsupported_claims": [],
            "abstention_correct": None,
        }
        result.update(overrides)
        return result

    def test_validates_scores_and_unsupported_claims(self):
        result = evaluator.validate_judge_result(self.valid_judge(), False)
        self.assertEqual(result["correctness_score"], 0.9)

    def test_rejects_malformed_score_without_assigning_a_default(self):
        with self.assertRaisesRegex(ValueError, "correctness_score"):
            evaluator.validate_judge_result(self.valid_judge(correctness_score="yes"), False)

    def test_requires_abstention_decision_for_unanswerable_question(self):
        with self.assertRaisesRegex(ValueError, "abstention_correct"):
            evaluator.validate_judge_result(self.valid_judge(), True)

    def test_rejects_api_failure(self):
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", side_effect=RuntimeError("offline")
        ):
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                evaluator.request_judge("prompt")

    def test_request_uses_openrouter_model_and_structured_response(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"ok": true}'),
                    finish_reason="stop",
                )
            ]
        )
        client = MagicMock()
        client.__enter__.return_value = client
        mocked_send = client.chat.send
        mocked_send.return_value = response
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", return_value=client
        ) as mocked_client:
            result = evaluator.request_judge("prompt")

        self.assertEqual(result, {"ok": True})
        mocked_client.assert_called_once_with(
            api_key="test-key", timeout_ms=evaluator.JUDGE_TIMEOUT_SECONDS * 1000
        )
        kwargs = mocked_send.call_args.kwargs
        self.assertEqual(kwargs["model"], evaluator.OPENROUTER_MODEL)
        self.assertEqual(kwargs["max_tokens"], evaluator.JUDGE_MAX_TOKENS)
        self.assertEqual(kwargs["stream"], False)
        self.assertEqual(kwargs["messages"][1]["content"], "prompt")

    def test_gemini_provider_uses_configured_model_and_validates_json(self):
        valid = self.valid_judge()
        response = SimpleNamespace(content=json.dumps(valid), text=json.dumps(valid))
        client = MagicMock()
        client.models.generate_content.return_value = response
        with patch.object(evaluator, "JUDGE_PROVIDER", "gemini"), patch.dict(
            "os.environ", {evaluator.GEMINI_API_KEY_ENV: "test-gemini-key"}
        ), patch.object(evaluator.genai, "Client", return_value=client) as mocked_client:
            result = evaluator.request_and_validate_judge("prompt", False)

        self.assertEqual(result["correctness_score"], 0.9)
        mocked_client.assert_called_once_with(api_key="test-gemini-key")
        kwargs = client.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["model"], evaluator.JUDGE_MODEL)
        self.assertEqual(kwargs["config"]["response_mime_type"], "application/json")

    def test_aion_provider_uses_configured_base_url_model_and_validates_json(self):
        valid = self.valid_judge()
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(valid)),
                )
            ]
        )
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch.object(evaluator, "JUDGE_PROVIDER", "aion"), patch.dict(
            "os.environ",
            {
                evaluator.AION_API_KEY_ENV: "test-aion-key",
                evaluator.AION_BASE_URL_ENV: "https://aion.example/v1",
            },
        ), patch.object(evaluator, "OpenAI", return_value=client) as mocked_client:
            result = evaluator.request_and_validate_judge("prompt", False)

        self.assertEqual(result["faithfulness_score"], 0.8)
        mocked_client.assert_called_once_with(
            api_key="test-aion-key",
            base_url="https://aion.example/v1",
            timeout=evaluator.JUDGE_TIMEOUT_SECONDS,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], evaluator.JUDGE_MODEL)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})

    def test_aion_authentication_failure_is_not_retried(self):
        with patch.object(evaluator, "JUDGE_PROVIDER", "aion"), patch.dict(
            "os.environ",
            {
                evaluator.AION_API_KEY_ENV: "test-aion-key",
                evaluator.AION_BASE_URL_ENV: "https://aion.example/v1",
            },
        ), patch.object(evaluator, "OpenAI", side_effect=RuntimeError("HTTP 401")) as client:
            with self.assertRaisesRegex(RuntimeError, "Aion authentication failed"):
                evaluator.request_and_validate_judge("prompt", False)
        client.assert_called_once()

    def test_rejects_client_authentication_failure(self):
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", side_effect=RuntimeError("HTTP 401")
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                evaluator.request_judge("prompt")

    def test_rejects_malformed_client_response(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="not json"),
                    finish_reason="stop",
                )
            ]
        )
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.send.return_value = response
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", return_value=client
        ):
            with self.assertRaisesRegex(ValueError, "valid JSON"):
                evaluator.request_judge("prompt")

    def test_retries_malformed_response_once_then_validates(self):
        valid = self.valid_judge()
        with patch.object(
            evaluator,
            "request_judge",
            side_effect=[ValueError("malformed JSON"), valid],
        ) as request:
            result = evaluator.request_and_validate_judge("prompt", False)
        self.assertEqual(result["correctness_score"], 0.9)
        self.assertEqual(request.call_count, 2)

    def test_retry_exhaustion_does_not_assign_scores(self):
        with patch.object(
            evaluator,
            "request_judge",
            side_effect=ValueError("malformed JSON"),
        ) as request:
            with self.assertRaisesRegex(ValueError, "malformed JSON"):
                evaluator.request_and_validate_judge("prompt", False)
        self.assertEqual(request.call_count, evaluator.JUDGE_RESPONSE_RETRY_LIMIT + 1)

    def test_rejects_missing_client_fields(self):
        response = SimpleNamespace(choices=[])
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.send.return_value = response
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", return_value=client
        ):
            with self.assertRaisesRegex(ValueError, "choices"):
                evaluator.request_judge("prompt")

    def test_missing_api_key_is_reported_without_a_request(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(
            evaluator, "OpenRouter"
        ) as mocked_client:
            with self.assertRaisesRegex(PermissionError, "OPENROUTER_API_KEY"):
                evaluator.request_judge("prompt")
        mocked_client.assert_not_called()

    def test_rejects_empty_content(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=""),
                    finish_reason="stop",
                )
            ]
        )
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.send.return_value = response
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", return_value=client
        ):
            with self.assertRaisesRegex(ValueError, "empty message content"):
                evaluator.request_judge("prompt")

    def test_rejects_incomplete_generation(self):
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"correctness_score":'),
                    finish_reason="length",
                )
            ]
        )
        client = MagicMock()
        client.__enter__.return_value = client
        client.chat.send.return_value = response
        with patch.dict("os.environ", {evaluator.OPENROUTER_API_KEY_ENV: "test-key"}), patch.object(
            evaluator, "OpenRouter", return_value=client
        ):
            with self.assertRaisesRegex(ValueError, "incomplete"):
                evaluator.request_judge("prompt")


class GenerationMetricTests(unittest.TestCase):
    def test_aggregate_scores_averages_valid_judgements(self):
        results = [
            {"judge": {"correctness_score": 1, "faithfulness_score": 0.5, "relevance_score": 1}},
            {"judge": {"correctness_score": 0, "faithfulness_score": 1, "relevance_score": 0.5}},
        ]
        self.assertEqual(
            evaluator.calculate_aggregate_scores(results),
            {"correctness": 0.5, "faithfulness": 0.75, "relevance": 0.75},
        )

    def test_empty_aggregate_is_zero(self):
        self.assertEqual(
            evaluator.calculate_aggregate_scores([]),
            {"correctness": 0.0, "faithfulness": 0.0, "relevance": 0.0},
        )


class GenerationExecutionTests(unittest.TestCase):
    def test_evaluate_captures_answers_context_and_incorrect_abstention(self):
        records = [
            {
                "id": "correct",
                "question": "What is answer one?",
                "expected_answer": "Answer one.",
                "expected_behavior": "Answer using the trusted context.",
                "trusted_context": [{"text": "Answer one."}],
            },
            {
                "id": "incorrect-abstention",
                "question": "What exact accuracy?",
                "question_category": "answer_absent",
                "expected_answer": None,
                "expected_behavior": "State that the available context does not contain enough information.",
                "trusted_context": [{"text": "No accuracy is provided."}],
            },
        ]
        judge_results = iter(
            [
                {
                    "correctness_score": 1,
                    "faithfulness_score": 1,
                    "relevance_score": 1,
                    "explanation": "Correct.",
                    "unsupported_claims": [],
                    "abstention_correct": None,
                },
                {
                    "correctness_score": 0,
                    "faithfulness_score": 0,
                    "relevance_score": 0.2,
                    "explanation": "It answered instead of abstaining.",
                    "unsupported_claims": ["The exact accuracy is 95%"],
                    "abstention_correct": False,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            golden_path = Path(directory) / "goldens.json"
            report_path = Path(directory) / "reports" / "generation.json"
            golden_path.write_text(json.dumps(records), encoding="utf-8")
            with (
                patch.object(evaluator.rag_chat, "load_index_and_metadata", return_value=(object(), [{"chunk_id": "chunk_1"}])),
                patch.object(evaluator, "SentenceTransformer", return_value=object()),
                patch.object(evaluator.rag_chat, "retrieve_chunks", return_value=[{"chunk_id": "chunk_1", "text": "context", "score": 0.9}]),
                patch.object(evaluator.rag_chat, "generate_answer", side_effect=["Answer one.", "The accuracy was 95%."]),
                patch.object(evaluator, "request_judge", side_effect=lambda prompt: next(judge_results)),
            ):
                report = evaluator.evaluate(golden_path, report_path)
            self.assertTrue(report_path.is_file())

        self.assertEqual(report["golden_records"], 2)
        self.assertEqual(report["evaluated_questions"], 2)
        self.assertEqual(report["failed_question_count"], 1)
        self.assertEqual(report["results"][0]["generated_answer"], "Answer one.")
        self.assertEqual(report["results"][1]["evaluation_status"], "failed")
        self.assertEqual(report["results"][0]["retrieved_context"][0]["chunk_id"], "chunk_1")


if __name__ == "__main__":
    unittest.main()
