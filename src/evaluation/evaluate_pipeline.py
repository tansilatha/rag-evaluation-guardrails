import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.evaluation import evaluate_generation, evaluate_retrieval
from src.rag import rag_chat


PROJECT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_DIR / "datasets" / "goldens" / "pipeline_goldens.json"
REPORT_PATH = PROJECT_DIR / "reports" / "pipeline_evaluation.json"
RETRIEVAL_K = rag_chat.TOP_K


def load_pipeline_goldens(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline golden dataset not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Pipeline golden dataset must be a JSON array.")
    return records


def prepare_records(
    records: list[dict[str, Any]], available_chunk_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared = []
    skipped = []
    for position, record in enumerate(records, start=1):
        record_id = str(record.get("id", f"record_{position}")) if isinstance(record, dict) else f"record_{position}"
        if not isinstance(record, dict):
            skipped.append({"id": record_id, "reason": "Record is not a JSON object."})
            continue
        question = record.get("question")
        relevant_ids = record.get("relevant_chunk_ids")
        expected_behavior = record.get("expected_behavior")
        if not isinstance(question, str) or not question.strip():
            skipped.append({"id": record_id, "reason": "Question is missing or empty."})
            continue
        if not isinstance(relevant_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in relevant_ids
        ):
            skipped.append({"id": record_id, "reason": "relevant_chunk_ids must be a list of non-empty strings."})
            continue
        missing_ids = sorted(set(relevant_ids) - available_chunk_ids)
        if missing_ids:
            raise ValueError(f"Pipeline golden {record_id} references unknown chunks: {', '.join(missing_ids)}")
        if not isinstance(expected_behavior, str) or not expected_behavior.strip():
            skipped.append({"id": record_id, "reason": "expected_behavior is missing or empty."})
            continue
        prepared.append(
            {
                "id": record_id,
                "question": question.strip(),
                "relevant_chunk_ids": list(dict.fromkeys(relevant_ids)),
                "expected_answer": record.get("expected_answer"),
                "expected_behavior": expected_behavior.strip(),
            }
        )
    return prepared, skipped


def evaluate(
    golden_path: Path = GOLDEN_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    raw_records = load_pipeline_goldens(golden_path)
    index, metadata = rag_chat.load_index_and_metadata()
    available_chunk_ids = {str(record["chunk_id"]) for record in metadata}
    records, skipped = prepare_records(raw_records, available_chunk_ids)
    model = SentenceTransformer(rag_chat.MODEL_NAME)
    results = []
    failures = []
    retrieval_totals = {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}
    retrieval_metric_key = f"retrieval_metrics_at_{RETRIEVAL_K}"
    end_to_end_latencies = []
    judge_latencies = []

    for item in records:
        started = time.perf_counter()
        try:
            chunks = rag_chat.retrieve_chunks(item["question"], model, index, metadata)
            retrieved_ids = [str(chunk["chunk_id"]) for chunk in chunks]
            retrieval_applicable = bool(item["relevant_chunk_ids"])
            retrieval_metrics = evaluate_retrieval.calculate_metrics(
                retrieved_ids, set(item["relevant_chunk_ids"]), RETRIEVAL_K
            )
            answer = rag_chat.generate_answer(item["question"], chunks)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Generator returned an empty answer.")
        except Exception as error:
            failures.append({"id": item["id"], "stage": "pipeline", "error": str(error)})
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "evaluation_status": "pipeline_error",
                    "error": str(error),
                }
            )
            continue
        end_to_end_latency = (time.perf_counter() - started) * 1000
        end_to_end_latencies.append(end_to_end_latency)
        if retrieval_applicable:
            for name, value in retrieval_metrics.items():
                retrieval_totals[name] += value

        judge_started = time.perf_counter()
        try:
            judge = evaluate_generation.request_and_validate_judge(
                evaluate_generation.judge_prompt(
                    item["question"],
                    answer,
                    chunks,
                    item["expected_answer"],
                    item["expected_behavior"],
                ),
                item["expected_answer"] is None,
            )
        except Exception as error:
            failures.append({"id": item["id"], "stage": "judge", "error": str(error)})
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "retrieved_chunk_ids": retrieved_ids,
                    "relevant_chunk_ids": item["relevant_chunk_ids"],
                    retrieval_metric_key: retrieval_metrics if retrieval_applicable else None,
                    "retrieval_metrics_applicable": retrieval_applicable,
                    "generated_answer": answer,
                    "end_to_end_latency_ms": end_to_end_latency,
                    "evaluation_status": "judge_error",
                    "error": str(error),
                }
            )
            continue
        judge_latencies.append((time.perf_counter() - judge_started) * 1000)
        result = {
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_behavior": item["expected_behavior"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": item["relevant_chunk_ids"],
            retrieval_metric_key: retrieval_metrics if retrieval_applicable else None,
            "retrieval_metrics_applicable": retrieval_applicable,
            "retrieved_context": chunks,
            "generated_answer": answer,
            "end_to_end_latency_ms": end_to_end_latency,
            "judge_latency_ms": judge_latencies[-1],
            "judge": judge,
            "evaluation_status": "evaluated",
        }
        if item["expected_answer"] is None and judge["abstention_correct"] is False:
            result["evaluation_status"] = "incorrect_answer"
            failures.append({"id": item["id"], "stage": "answer", "error": "Incorrect answer for an unanswerable question."})
        results.append(result)

    evaluated_count = len(results)
    scored_count = sum(isinstance(result.get("judge"), dict) for result in results)
    retrieval_evaluated_count = sum(
        result.get("retrieval_metrics_applicable", False) for result in results
    )
    report = {
        "golden_records": len(raw_records),
        "evaluated_questions": evaluated_count,
        "scored_questions": scored_count,
        "skipped_question_count": len(skipped),
        "failed_question_count": len(failures),
        "skipped_questions": skipped,
        "failures": failures,
        f"retrieval_success_rate_at_{RETRIEVAL_K}": (
            sum(
                result.get(retrieval_metric_key, {}).get("hit_rate", 0.0)
                for result in results
                if result.get("retrieval_metrics_applicable")
            )
            / retrieval_evaluated_count if retrieval_evaluated_count else 0.0
        ),
        "retrieval_evaluated_questions": retrieval_evaluated_count,
        f"retrieval_metrics_at_{RETRIEVAL_K}": {
            name: retrieval_totals[name] / retrieval_evaluated_count if retrieval_evaluated_count else 0.0
            for name in retrieval_totals
        },
        "answer_scores": evaluate_generation.calculate_aggregate_scores(results),
        "average_end_to_end_latency_ms": statistics.mean(end_to_end_latencies) if end_to_end_latencies else 0.0,
        "average_judge_latency_ms": statistics.mean(judge_latencies) if judge_latencies else 0.0,
        "pipeline_configuration": {
            "retriever": "rag_chat.retrieve_chunks",
            "context_and_generator": "rag_chat.generate_answer",
            "embedding_model": rag_chat.MODEL_NAME,
            "generator_model": rag_chat.OLLAMA_MODEL,
            "retrieval_top_k": rag_chat.TOP_K,
            "relevance_field": "relevant_chunk_ids",
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated questions: {evaluated_count}")
    print(f"Scored questions: {scored_count}")
    print(f"Skipped questions: {len(skipped)}")
    print(f"Failed questions: {len(failures)}")
    print(f"Retrieval success rate@{RETRIEVAL_K}: {report[f'retrieval_success_rate_at_{RETRIEVAL_K}']:.3f}")
    print(f"Average end-to-end latency: {report['average_end_to_end_latency_ms']:.2f} ms")
    print(f"Report saved to: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the complete RAG pipeline.")
    parser.add_argument("--goldens", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    evaluate(args.goldens, args.report)


if __name__ == "__main__":
    main()
