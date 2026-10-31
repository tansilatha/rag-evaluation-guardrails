import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.rag.retrieve import (
    MODEL_NAME,
    embed_question,
    load_index_and_metadata,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_DIR / "datasets" / "goldens" / "retrieval_goldens.json"
REPORT_PATH = PROJECT_DIR / "reports" / "retrieval_evaluation.json"
EVALUATION_K_VALUES = (1, 3, 5)


def load_golden_questions(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval golden dataset not found: {path}")

    try:
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Golden dataset contains invalid JSON: {error}") from error

    if not isinstance(records, list):
        raise ValueError("Retrieval golden dataset must be a JSON array.")
    return records


def is_explicitly_unanswerable(record: dict[str, Any]) -> bool:
    if record.get("answerable") is False:
        return True
    if record.get("question_category") == "answer_absent":
        return True

    expected_behavior = record.get("expected_behavior")
    return (
        isinstance(expected_behavior, str)
        and "not contain enough information" in expected_behavior.casefold()
    )


def prepare_evaluation_records(
    records: list[dict[str, Any]],
    available_chunk_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    evaluated = []
    skipped = []

    for position, record in enumerate(records, start=1):
        record_id = (
            str(record.get("id", f"record_{position}"))
            if isinstance(record, dict)
            else f"record_{position}"
        )
        if not isinstance(record, dict):
            skipped.append(
                {"id": record_id, "reason": "Record is not a JSON object."}
            )
            continue
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            skipped.append(
                {"id": record_id, "reason": "Question is missing or empty."}
            )
            continue
        if "relevant_chunk_ids" not in record:
            skipped.append(
                {"id": record_id, "reason": "Missing relevant_chunk_ids label."}
            )
            continue

        relevant_ids = record["relevant_chunk_ids"]
        if not isinstance(relevant_ids, list) or any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            skipped.append(
                {
                    "id": record_id,
                    "reason": "relevant_chunk_ids must be a list of non-empty strings.",
                }
            )
            continue
        if not relevant_ids:
            reason = (
                "Unanswerable question has no relevant chunks."
                if is_explicitly_unanswerable(record)
                else "Empty relevance labels; answerability is not explicit."
            )
            skipped.append({"id": record_id, "reason": reason})
            continue

        missing_ids = sorted(set(relevant_ids) - available_chunk_ids)
        if missing_ids:
            raise ValueError(
                f"Golden {record_id} references chunk IDs absent from FAISS "
                f"metadata: {', '.join(missing_ids)}."
            )

        evaluated.append(
            {
                "id": record_id,
                "question": question.strip(),
                "relevant_chunk_ids": list(dict.fromkeys(relevant_ids)),
            }
        )

    return evaluated, skipped


def calculate_metrics(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("K must be a positive integer.")

    unique_retrieved = list(dict.fromkeys(retrieved_chunk_ids))[:k]
    relevant_count = sum(
        chunk_id in relevant_chunk_ids for chunk_id in unique_retrieved
    )
    precision = relevant_count / k
    recall = relevant_count / len(relevant_chunk_ids) if relevant_chunk_ids else 0.0
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, chunk_id in enumerate(unique_retrieved, start=1)
            if chunk_id in relevant_chunk_ids
        ),
        0.0,
    )
    return {
        "hit_rate": 1.0 if relevant_count else 0.0,
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
    }


def retrieve_chunk_ids(
    query: Any,
    index: Any,
    metadata: list[dict[str, Any]],
    top_k: int,
) -> list[str]:
    _, positions = index.search(query, top_k)
    retrieved = []
    seen = set()
    for raw_position in positions[0]:
        position = int(raw_position)
        if position == -1:
            continue
        if position < 0 or position >= len(metadata):
            raise ValueError(
                f"FAISS returned invalid vector position {position}."
            )
        chunk_id = metadata[position].get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                f"FAISS metadata at position {position} has no valid chunk_id."
            )
        if chunk_id not in seen:
            retrieved.append(chunk_id)
            seen.add(chunk_id)
    return retrieved


def evaluate(
    golden_path: Path = GOLDEN_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    raw_records = load_golden_questions(golden_path)
    index, metadata = load_index_and_metadata()
    available_chunk_ids = {
        str(record["chunk_id"]) for record in metadata
    }
    questions, skipped = prepare_evaluation_records(
        raw_records, available_chunk_ids
    )

    model = SentenceTransformer(MODEL_NAME)
    max_k = min(max(EVALUATION_K_VALUES), int(index.ntotal))
    totals = {
        k: {"hit_rate": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0}
        for k in EVALUATION_K_VALUES
    }
    question_results = []
    latencies_ms = []

    for item in questions:
        started = time.perf_counter()
        query = embed_question(model, item["question"])
        retrieved_ids = retrieve_chunk_ids(query, index, metadata, max_k)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)

        per_k = {}
        relevant_ids = set(item["relevant_chunk_ids"])
        for k in EVALUATION_K_VALUES:
            metrics = calculate_metrics(retrieved_ids, relevant_ids, k)
            per_k[str(k)] = metrics
            for name, value in metrics.items():
                totals[k][name] += value

        question_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": item["relevant_chunk_ids"],
                "retrieval_latency_ms": latency_ms,
                "metrics": per_k,
            }
        )

    question_count = len(questions)
    metrics_by_k = {
        str(k): {
            name: totals[k][name] / question_count if question_count else 0.0
            for name in ("hit_rate", "recall", "precision", "mrr")
        }
        for k in EVALUATION_K_VALUES
    }
    report = {
        "evaluated_questions": question_count,
        "skipped_question_count": len(skipped),
        "skipped_questions": skipped,
        "metrics_by_k": metrics_by_k,
        "average_retrieval_latency_ms": (
            statistics.mean(latencies_ms) if latencies_ms else 0.0
        ),
        "latency_definition": (
            "Elapsed time for query embedding and FAISS search at depth "
            f"{max_k}; excludes model/index loading and report formatting."
        ),
        "retriever_configuration": {
            "implementation": "FAISS search via retrieve.load_index_and_metadata",
            "index_path": str(
                (PROJECT_DIR / "data" / "faiss.index").resolve()
            ),
            "index_type": type(index).__name__,
            "index_vectors": int(index.ntotal),
            "index_dimensions": int(index.d),
            "index_metric": "inner product over normalized vectors (cosine similarity)",
            "searched_top_k": max_k,
            "embedding_model": MODEL_NAME,
            "embedding_dimension": 384,
            "relevance_field": "relevant_chunk_ids",
        },
        "duplicate_handling": (
            "Duplicate chunk IDs are removed while preserving first rank before "
            "calculating metrics. Precision is divided by requested K."
        ),
        "empty_retrieval_handling": (
            "Empty retrieval produces zero for hit rate, recall, precision, and MRR."
        ),
        "question_results": question_results,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(report, report_path)
    return report


def print_report(report: dict[str, Any], report_path: Path) -> None:
    print(f"Evaluated questions: {report['evaluated_questions']}")
    print(f"Skipped questions: {len(report['skipped_questions'])}")
    for k, metrics in report["metrics_by_k"].items():
        print(
            f"@{k}: Hit Rate={metrics['hit_rate']:.3f}, "
            f"Recall={metrics['recall']:.3f}, "
            f"Precision={metrics['precision']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )
    print(
        "Average retrieval latency: "
        f"{report['average_retrieval_latency_ms']:.2f} ms"
    )
    for skipped in report["skipped_questions"]:
        print(f"Skipped {skipped['id']}: {skipped['reason']}")
    print(f"Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the existing FAISS retriever against retrieval goldens."
    )
    parser.add_argument(
        "--goldens",
        type=Path,
        default=GOLDEN_PATH,
        help="retrieval golden dataset JSON path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORT_PATH,
        help="output report JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        evaluate(args.goldens, args.report)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Retrieval evaluation error: {error}") from error


if __name__ == "__main__":
    main()
