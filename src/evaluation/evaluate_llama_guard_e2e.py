import argparse
import json
import time
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.evaluation import evaluate_llama_guard
from src.rag import rag_chat
from src.rag.retrieve import MODEL_NAME, load_index_and_metadata


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = evaluate_llama_guard.DATASET_PATH
REPORT_DIR = PROJECT_DIR / "reports" / "llama_guard"
JSON_REPORT_PATH = REPORT_DIR / "end_to_end_safety_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "end_to_end_safety_evaluation.md"


def evaluate(
    records: list[dict[str, Any]],
    model: Any,
    index: Any,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise ValueError("The Llama Guard end-to-end dataset is empty.")

    results = []
    for record in records:
        started = time.perf_counter()
        try:
            pipeline_result = rag_chat.run_guarded_rag(
                record["text"], model, index, metadata
            )
            input_prediction, input_status, input_reason = (
                evaluate_llama_guard.classify_result(
                    (pipeline_result["input_safe"], pipeline_result["input_reason"])
                )
            )
            status = pipeline_result["status"]
            pipeline_error = pipeline_result.get("error")
            output_prediction = None
            output_status = "not_checked"
            output_reason = None
            if status in {"completed", "output_blocked"}:
                output_prediction, output_status, output_reason = (
                    evaluate_llama_guard.classify_result(
                        (
                            pipeline_result["output_safe"],
                            pipeline_result["output_reason"],
                        )
                    )
                )
        except (ConnectionError, RuntimeError, TimeoutError, ValueError, KeyError) as error:
            status = "error"
            pipeline_error = f"{type(error).__name__}: {error}"
            input_prediction = None
            input_status = "error"
            input_reason = pipeline_error
            output_prediction = None
            output_status = "not_checked"
            output_reason = None

        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            {
                "id": record["id"],
                "category": record["category"],
                "expected_input_label": record["expected_label"],
                "actual_input_prediction": input_prediction,
                "input_status": input_status,
                "input_reason": input_reason,
                "output_prediction": output_prediction,
                "output_status": output_status,
                "output_reason": output_reason,
                "pipeline_status": status,
                "pipeline_error": pipeline_error,
                "latency_ms": latency_ms,
            }
        )

    input_records = [
        {
            "expected_label": result["expected_input_label"],
            "actual_prediction": result["actual_input_prediction"],
        }
        for result in results
    ]
    output_checked = [
        result
        for result in results
        if result["output_status"] in {"classified", "unexpected_response", "error"}
    ]
    completed = sum(result["pipeline_status"] == "completed" for result in results)
    input_blocked = sum(
        result["pipeline_status"] == "input_blocked" for result in results
    )
    output_blocked = sum(
        result["pipeline_status"] == "output_blocked" for result in results
    )
    errors = [result for result in results if result["pipeline_status"] == "error"]
    input_guard_errors = sum(
        result["input_status"] == "error" for result in results
    )
    input_guard_unexpected = sum(
        result["input_status"] == "unexpected_response" for result in results
    )
    output_guard_errors = sum(
        result["output_status"] == "error" for result in results
    )
    output_guard_unexpected = sum(
        result["output_status"] == "unexpected_response" for result in results
    )
    unsafe_inputs = [
        result for result in results if result["expected_input_label"] == "UNSAFE"
    ]
    scored_unsafe_inputs = [
        result
        for result in unsafe_inputs
        if result["input_status"] == "classified"
    ]
    missed_unsafe_inputs = sum(
        result["actual_input_prediction"] != "UNSAFE"
        for result in scored_unsafe_inputs
    )

    return {
        "dataset": str(DATASET_PATH),
        "evaluation": "end_to_end_rag_safety",
        "guard_configuration": {
            "guard_model": evaluate_llama_guard.llama_guard.LLAMA_GUARD_MODEL,
            "guard_url": evaluate_llama_guard.llama_guard.LLAMA_GUARD_URL,
            "rag_model": rag_chat.OLLAMA_MODEL,
        },
        "total_cases": len(results),
        "input_classification_metrics": evaluate_llama_guard.calculate_metrics(
            input_records
        ),
        "pipeline_outcomes": {
            "completed": completed,
            "input_blocked": input_blocked,
            "output_blocked": output_blocked,
            "errors": len(errors),
            "input_guard_errors": input_guard_errors,
            "input_guard_unexpected_responses": input_guard_unexpected,
            "output_guard_errors": output_guard_errors,
            "output_guard_unexpected_responses": output_guard_unexpected,
            "unsafe_input_count": len(unsafe_inputs),
            "unsafe_input_unscored_count": len(unsafe_inputs)
            - len(scored_unsafe_inputs),
            "unsafe_input_missed_count": missed_unsafe_inputs,
            "unsafe_input_block_rate": (
                (len(scored_unsafe_inputs) - missed_unsafe_inputs)
                / len(scored_unsafe_inputs)
                if scored_unsafe_inputs
                else 0.0
            ),
            "output_checks_performed": len(output_checked),
            "output_safe": sum(
                result["output_prediction"] == "SAFE" for result in results
            ),
            "output_unsafe_blocked": sum(
                result["pipeline_status"] == "output_blocked"
                and result["output_prediction"] == "UNSAFE"
                for result in results
            ),
            "output_labels_available": False,
            "output_label_note": (
                "End-to-end output classification metrics require human labels "
                "for the actual generated answers. Use the separate output "
                "evaluation dataset for labeled output metrics."
            ),
        },
        "average_latency_ms": sum(result["latency_ms"] for result in results)
        / len(results),
        "errors": [
            {
                "id": result["id"],
                "reason": result["pipeline_error"]
                or result["input_reason"]
                or result["output_reason"],
            }
            for result in errors
        ],
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["input_classification_metrics"]
    outcomes = report["pipeline_outcomes"]
    lines = [
        "# Llama Guard End-to-End RAG Safety Evaluation",
        "",
        f"- Cases: {report['total_cases']}",
        f"- Average pipeline latency: {report['average_latency_ms']:.2f} ms",
        f"- Input accuracy: {metrics['accuracy']:.4f}",
        f"- Input precision (UNSAFE): {metrics['precision']:.4f}",
        f"- Input recall (UNSAFE): {metrics['recall']:.4f}",
        f"- Input F1-score: {metrics['f1_score']:.4f}",
        f"- Input false positives: {metrics['false_positives']}",
        f"- Input false negatives: {metrics['false_negatives']}",
        f"- Input confusion matrix: {metrics['confusion_matrix']}",
        "",
        "## Pipeline enforcement",
        "",
        f"- Completed: {outcomes['completed']}",
        f"- Blocked before retrieval: {outcomes['input_blocked']}",
        f"- Blocked before answer delivery: {outcomes['output_blocked']}",
        f"- Errors: {outcomes['errors']}",
        f"- Unsafe input block rate: {outcomes['unsafe_input_block_rate']:.4f}",
        f"- Output checks performed: {outcomes['output_checks_performed']}",
        f"- Unsafe outputs blocked: {outcomes['output_unsafe_blocked']}",
        "",
        f"**Output label limitation:** {outcomes['output_label_note']}",
        "",
        "## Per-case outcomes",
        "",
        "| ID | Expected input | Actual input | Input status | Output prediction | Pipeline status | Latency (ms) |",
        "|---|---|---|---|---|---|---:|",
    ]
    for result in report["results"]:
        lines.append(
            f"| {result['id']} | {result['expected_input_label']} "
            f"| {result['actual_input_prediction'] or '—'} "
            f"| {result['input_status']} "
            f"| {result['output_prediction'] or '—'} "
            f"| {result['pipeline_status']} | {result['latency_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_reports(
    report: dict[str, Any],
    json_report_path: Path,
    markdown_report_path: Path,
) -> None:
    if json_report_path.resolve() == markdown_report_path.resolve():
        raise ValueError("JSON and Markdown report paths must be different.")
    existing = [
        path
        for path in (json_report_path, markdown_report_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing report(s): "
            + ", ".join(str(path) for path in existing)
        )
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_report_path.write_text(render_markdown(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Llama Guard enforcement in the RAG pipeline."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json-report", type=Path, default=JSON_REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = evaluate_llama_guard.load_dataset(args.dataset)
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer(MODEL_NAME)
    report = evaluate(records, model, index, metadata)
    report["dataset"] = str(args.dataset.resolve())
    write_reports(report, args.json_report, args.markdown_report)
    print(f"End-to-end safety evaluation complete: {report['total_cases']} cases")
    print(f"Input accuracy: {report['input_classification_metrics']['accuracy']:.4f}")
    print(f"Input F1-score: {report['input_classification_metrics']['f1_score']:.4f}")
    print(f"Average pipeline latency: {report['average_latency_ms']:.2f} ms")
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.markdown_report}")


if __name__ == "__main__":
    main()
