import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from src.evaluation import golden_dataset_builder as builder


class GoldenDatasetBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.source = builder.load_source_document()
        self.chunks = builder.load_existing_chunks()

        for name in (
            "GOLDEN_DIR",
            "RETRIEVAL_PATH",
            "GENERATION_PATH",
            "PIPELINE_PATH",
            "CANDIDATES_PATH",
            "REJECTED_PATH",
        ):
            filename = {
                "GOLDEN_DIR": "",
                "RETRIEVAL_PATH": "retrieval.json",
                "GENERATION_PATH": "generation.json",
                "PIPELINE_PATH": "pipeline.json",
                "CANDIDATES_PATH": "candidates.jsonl",
                "REJECTED_PATH": "rejected.jsonl",
            }[name]
            path = self.output_dir if not filename else self.output_dir / filename
            patcher = patch.object(builder, name, path)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def valid_candidate(self, **overrides):
        supporting_passage = (
            "Supervised learning is a machine learning approach in which a model "
            "learns from labeled training data."
        )
        candidate = {
            "id": "golden_0001",
            "question": "What does supervised learning use?",
            "question_category": "simple_factual",
            "answerable": True,
            "candidate_expected_answer": "Labeled training data.",
            "supporting_passages": [supporting_passage],
            "supporting_evidence": [
                {
                    "passage": supporting_passage,
                    "chunk_id": "chunk_0001",
                    "source": "machine_learning.txt",
                    "text": self.chunks[1]["text"],
                }
            ],
            "relevant_chunk_ids": ["chunk_0001"],
            "expected_behavior": "Answer using the trusted context.",
            "review_status": "pending",
        }
        candidate.update(overrides)
        return candidate

    def test_source_chunks_and_chunk_id_validation(self):
        self.assertTrue(self.source)
        self.assertEqual(len(self.chunks), 19)
        self.assertEqual(builder.validate_chunk_ids(["chunk_0001"], self.chunks), [])
        self.assertEqual(
            builder.validate_chunk_ids(["missing_chunk"], self.chunks),
            ["missing_chunk"],
        )

    def test_evidence_is_traced_to_actual_source_chunk(self):
        evidence, errors = builder.extract_supporting_evidence(
            {
                "supporting_passages": [
                    "Each training example contains input features and a known target or output."
                ]
            },
            self.source,
            self.chunks,
        )
        self.assertFalse(errors)
        self.assertEqual([item["chunk_id"] for item in evidence], ["chunk_0001"])
        self.assertIn(
            "Each training example contains input features",
            evidence[0]["text"],
        )

    def test_evidence_selection_keeps_multiple_chunks_for_comparisons(self):
        evidence = builder.select_relevant_chunks(
            "How do classification and regression differ in their outputs?",
            "Classification predicts discrete categories, while regression predicts continuous numerical values.",
            self.chunks,
        )
        chunk_ids = {item["chunk_id"] for item in evidence}
        self.assertIn("chunk_0002", chunk_ids)
        self.assertIn("chunk_0003", chunk_ids)

    def test_candidate_validation_and_relaxed_category_wording(self):
        candidate = self.valid_candidate(
            question="What are the two major categories of supervised learning?",
            question_category="comparison",
            candidate_expected_answer="Classification and regression.",
            supporting_passages=[
                "Supervised learning has two major categories: classification and regression."
            ],
        )
        checked, errors = builder.validate_candidate(
            candidate, self.source, self.chunks
        )
        self.assertFalse(errors)
        self.assertEqual(checked["relevant_chunk_ids"], ["chunk_0002"])

    def test_unanswerable_candidate_has_no_answer_or_relevant_ids(self):
        candidate = {
            "question": "What exact test accuracy did the model achieve?",
            "question_category": "answer_absent",
            "answerable": False,
            "candidate_expected_answer": "",
            "supporting_passages": [],
        }
        checked, errors = builder.validate_candidate(
            candidate, self.source, self.chunks
        )
        self.assertFalse(errors)
        self.assertEqual(checked["relevant_chunk_ids"], [])
        self.assertIn("not contain enough information", checked["expected_behavior"])

    def test_retry_on_rate_limit(self):
        rate_limit = HTTPError(
            "https://example.invalid",
            429,
            "rate limited",
            {},
            io.BytesIO(b""),
        )
        successful_response = io.BytesIO(b'{"ok": true}')
        with patch.object(
            builder,
            "urlopen",
            side_effect=[rate_limit, successful_response],
        ) as mocked_urlopen, patch.object(builder.time, "sleep"):
            result = builder.post_json(
                "https://example.invalid",
                {"test": True},
                {},
                1,
                "Ollama",
            )
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mocked_urlopen.call_count, 2)

    def test_non_retryable_http_error_is_reported(self):
        unauthorized = HTTPError(
            "https://example.invalid",
            401,
            "unauthorized",
            {},
            io.BytesIO(b""),
        )
        with patch.object(builder, "urlopen", side_effect=unauthorized):
            with self.assertRaisesRegex(RuntimeError, "Ollama request failed with HTTP 401"):
                builder.post_json(
                    "https://example.invalid",
                    {"test": True},
                    {},
                    1,
                    "Ollama",
                )

    def test_connection_errors_retry_then_fail(self):
        from urllib.error import URLError

        with patch.object(
            builder, "urlopen", side_effect=URLError("offline")
        ) as mocked_urlopen, patch.object(builder.time, "sleep"):
            with self.assertRaisesRegex(ConnectionError, "after 3 attempts"):
                builder.post_json(
                    "https://example.invalid",
                    {"test": True},
                    {},
                    1,
                    "Ollama",
                )
        self.assertEqual(mocked_urlopen.call_count, 3)

    def test_save_datasets_excludes_pending_and_rejected_from_goldens(self):
        accepted = self.valid_candidate(
            review_status="approved",
            source_document="machine_learning.txt",
        )
        pending = self.valid_candidate(
            id="golden_0002", review_status="pending"
        )
        rejected = self.valid_candidate(
            id="golden_0003", review_status="rejected"
        )
        builder.save_golden_datasets([accepted, pending, rejected])
        builder.save_candidate_logs([accepted, pending, rejected])

        retrieval = json.loads(builder.RETRIEVAL_PATH.read_text(encoding="utf-8"))
        generation = json.loads(builder.GENERATION_PATH.read_text(encoding="utf-8"))
        pipeline = json.loads(builder.PIPELINE_PATH.read_text(encoding="utf-8"))
        rejected_log = builder.load_jsonl_if_exists(builder.REJECTED_PATH)
        candidates_log = builder.load_jsonl_if_exists(builder.CANDIDATES_PATH)
        self.assertEqual([item["id"] for item in retrieval], ["golden_0001"])
        self.assertIn("supporting_evidence", retrieval[0])
        self.assertNotIn("expected_answer", retrieval[0])
        self.assertIn("relevant_chunk_ids", retrieval[0])
        self.assertIn("trusted_context", generation[0])
        self.assertNotIn("relevant_chunk_ids", generation[0])
        self.assertIn("source_evidence", generation[0])
        self.assertEqual(generation[0]["expected_answer"], "Labeled training data.")
        self.assertEqual(
            generation[0]["expected_behavior"], "Answer using the trusted context."
        )
        self.assertIn("trusted_evidence", pipeline[0])
        self.assertIn("relevant_chunk_ids", pipeline[0])
        self.assertIn("expected_answer", pipeline[0])
        self.assertEqual([item["id"] for item in rejected_log], ["golden_0003"])
        self.assertEqual(len(candidates_log), 3)

    def test_build_refuses_to_overwrite_existing_historical_goldens(self):
        for path in (
            builder.RETRIEVAL_PATH,
            builder.GENERATION_PATH,
            builder.PIPELINE_PATH,
        ):
            path.write_text("historical\n", encoding="utf-8")
        approved = self.valid_candidate(
            review_status="approved",
            source_document="machine_learning.txt",
        )
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            builder.save_golden_datasets([approved])

    def test_duplicate_question_is_not_added_again(self):
        old_candidate = self.valid_candidate()
        builder.write_jsonl(builder.CANDIDATES_PATH, [old_candidate])
        args = Namespace(test_mode=False, count=1)
        with patch.object(
            builder,
            "generate_candidate_questions",
            return_value=[
                {
                    "question": old_candidate["question"],
                    "question_category": "simple_factual",
                    "answerable": True,
                    "candidate_expected_answer": "answer",
                    "supporting_passages": old_candidate["supporting_passages"],
                }
            ],
        ), patch("builtins.print"):
            builder.run_candidate_generation(args)
        self.assertEqual(len(builder.load_candidate_log()), 1)

    def test_generation_requires_human_review_before_building_goldens(self):
        examples = [
            {
                "question": "What does supervised learning use?",
                "question_category": "simple_factual",
                "answerable": True,
                "candidate_expected_answer": "Labeled training data.",
                "supporting_passages": [
                    "Supervised learning is a machine learning approach in which a model learns from labeled training data."
                ],
            },
            {
                "question": "What exact test accuracy did the model achieve?",
                "question_category": "answer_absent",
                "answerable": False,
                "candidate_expected_answer": "",
                "supporting_passages": [],
            },
        ]
        args = Namespace(test_mode=True, count=10)
        with patch.object(
            builder, "generate_candidate_questions", return_value=examples
        ), patch("builtins.print"):
            builder.run_candidate_generation(args)
        candidates = builder.load_candidate_log()
        self.assertEqual([item["review_status"] for item in candidates], ["pending", "pending"])
        self.assertFalse(builder.RETRIEVAL_PATH.exists())
        self.assertFalse(builder.GENERATION_PATH.exists())
        self.assertFalse(builder.PIPELINE_PATH.exists())

        for candidate in candidates:
            candidate["review_status"] = "approved"
            candidate["review_notes"] = "Reviewed against source evidence."
        builder.write_jsonl(builder.CANDIDATES_PATH, candidates)
        builder.build_reviewed_goldens(overwrite=False)

        retrieval = json.loads(builder.RETRIEVAL_PATH.read_text(encoding="utf-8"))
        generation = json.loads(builder.GENERATION_PATH.read_text(encoding="utf-8"))
        pipeline = json.loads(builder.PIPELINE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(retrieval), 2)
        self.assertEqual(len(generation), 2)
        self.assertEqual(len(pipeline), 2)
        self.assertEqual(retrieval[1]["relevant_chunk_ids"], [])
        self.assertIsNone(generation[1]["expected_answer"])
        self.assertIn("not contain enough information", generation[1]["expected_behavior"])
        self.assertEqual(
            generation[1]["trusted_context"][0]["text"], self.source
        )
        self.assertEqual(pipeline[1]["relevant_chunk_ids"], [])
        self.assertIsNone(pipeline[1]["expected_answer"])
        self.assertIn("not contain enough information", pipeline[1]["expected_behavior"])
        self.assertTrue(all(item["review_status"] == "approved" for item in pipeline))

    def test_checked_in_golden_schemas_and_evidence_match_source(self):
        project_dir = Path(__file__).resolve().parents[1]
        data_dir = project_dir / "data"
        golden_dir = project_dir / "datasets" / "goldens"
        source = (data_dir / "documents" / "machine_learning.txt").read_text(
            encoding="utf-8"
        )
        chunk_lookup = {chunk["chunk_id"]: chunk for chunk in self.chunks}

        def load_dataset(name):
            with (golden_dir / name).open(encoding="utf-8") as file:
                return json.load(file)

        retrieval = load_dataset("retrieval_goldens.json")
        generation = load_dataset("generation_goldens.json")
        pipeline = load_dataset("pipeline_goldens.json")
        self.assertEqual(
            {item["id"] for item in retrieval},
            {item["id"] for item in generation},
        )
        self.assertEqual(
            {item["id"] for item in retrieval},
            {item["id"] for item in pipeline},
        )

        retrieval_by_id = {item["id"]: item for item in retrieval}
        generation_by_id = {item["id"]: item for item in generation}
        pipeline_by_id = {item["id"]: item for item in pipeline}
        for record_id in retrieval_by_id:
            retrieval_record = retrieval_by_id[record_id]
            generation_record = generation_by_id[record_id]
            pipeline_record = pipeline_by_id[record_id]
            self.assertEqual(retrieval_record["verification_status"], "accepted")
            self.assertEqual(generation_record["verification_status"], "accepted")
            self.assertEqual(pipeline_record["verification_status"], "accepted")
            self.assertTrue(retrieval_record["question"].strip())
            self.assertTrue(generation_record["question"].strip())
            self.assertTrue(pipeline_record["question"].strip())

            relevant_ids = retrieval_record["relevant_chunk_ids"]
            self.assertEqual(relevant_ids, pipeline_record["relevant_chunk_ids"])
            self.assertTrue(set(relevant_ids) <= set(chunk_lookup))
            self.assertNotIn("expected_answer", retrieval_record)
            self.assertNotIn("relevant_chunk_ids", generation_record)

            for evidence_key, record in (
                ("supporting_evidence", retrieval_record),
                ("source_evidence", generation_record),
                ("trusted_evidence", pipeline_record),
            ):
                evidence = record[evidence_key]
                evidence_ids = {item["chunk_id"] for item in evidence}
                self.assertEqual(evidence_ids, set(relevant_ids))
                for item in evidence:
                    self.assertIn(item["chunk_id"], chunk_lookup)
                    self.assertIn(
                        " ".join(item["passage"].split()).casefold(),
                        " ".join(source.split()).casefold(),
                    )
                    self.assertIn(
                        " ".join(item["passage"].split()).casefold(),
                        " ".join(
                            chunk_lookup[item["chunk_id"]]["text"].split()
                        ).casefold(),
                    )

            context = generation_record["trusted_context"]
            self.assertTrue(context)
            if generation_record["expected_answer"] is None:
                self.assertEqual(relevant_ids, [])
                self.assertEqual(generation_record["source_evidence"], [])
                self.assertEqual(
                    generation_record["expected_behavior"],
                    "State that the available context does not contain enough information.",
                )
                self.assertEqual(
                    pipeline_record["expected_behavior"],
                    generation_record["expected_behavior"],
                )
                self.assertEqual(context[0]["text"], source.strip())
                self.assertNotRegex(source, r"\b\d+(?:\.\d+)?\s*%")
            else:
                self.assertTrue(generation_record["expected_answer"].strip())
                self.assertEqual(
                    generation_record["expected_behavior"],
                    "Answer using the trusted context.",
                )
                self.assertEqual(
                    pipeline_record["expected_answer"],
                    generation_record["expected_answer"],
                )
                context_ids = {item["chunk_id"] for item in context}
                self.assertEqual(context_ids, set(relevant_ids))
                self.assertEqual(
                    " ".join(pipeline_record["expected_behavior"].split()),
                    "Answer using the trusted context.",
                )

    def test_approved_candidate_with_validation_errors_is_rejected(self):
        candidate = self.valid_candidate(
            review_status="approved",
            source_document="machine_learning.txt",
            validation_errors=["Evidence validation failed."],
        )
        with self.assertRaisesRegex(ValueError, "has validation errors"):
            builder.save_golden_datasets([candidate])

    def test_answerable_golden_answers_have_source_evidence_for_each_claim(self):
        golden_dir = (
            Path(__file__).resolve().parents[1] / "datasets" / "goldens"
        )
        with (golden_dir / "generation_goldens.json").open(
            encoding="utf-8"
        ) as file:
            generation = json.load(file)
        by_id = {record["id"]: record for record in generation}
        supported_claims = {
            "golden_0004": [
                "common supervised learning algorithms include linear regression, logistic regression, decision trees, random forests, support vector machines, and neural networks."
            ],
            "golden_0006": [
                "supervised learning has two major categories: classification and regression."
            ],
            "golden_0007": [
                "overfitting occurs when a machine learning model learns the training data too closely, including noise and accidental patterns.",
                "overfitting can be reduced through regularization, cross-validation, early stopping",
                "data augmentation",
            ],
            "golden_0008": [
                "the training set is used to fit the model.",
                "the validation set helps with model selection and hyperparameter tuning.",
                "the test set provides a final assessment on data not used during model development.",
            ],
            "golden_0010": [
                "a deep tree can overfit its training data.",
                "overfitting occurs when a machine learning model learns the training data too closely, including noise and accidental patterns.",
                "poorly on unseen data.",
            ],
        }
        for record_id, claims in supported_claims.items():
            record = by_id[record_id]
            evidence_text = " ".join(
                " ".join(item["passage"].split()).casefold()
                for item in record["source_evidence"]
            )
            for claim in claims:
                with self.subTest(record_id=record_id, claim=claim):
                    self.assertIn(claim, evidence_text)
            self.assertTrue(record["expected_answer"].strip())


if __name__ == "__main__":
    unittest.main()
