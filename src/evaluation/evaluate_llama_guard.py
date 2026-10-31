import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.guardrails import llama_guard


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_DIR / "datasets" / "llama_guard_evaluation.json"
OUTPUT_DATASET_PATH = PROJECT_DIR / "datasets" / "llama_guard_output_evaluation.json"
REPORT_DIR = PROJECT_DIR / "reports" / "llama_guard"
JSON_REPORT_PATH = REPORT_DIR / "input_safety_evaluation.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "input_safety_evaluation.md"
VALID_LABELS = {"SAFE", "UNSAFE"}


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Llama Guard dataset not found: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Llama Guard dataset contains invalid JSON: {error}") from error
    if not isinstance(records, list):
        raise ValueError("Llama Guard dataset must be a JSON array.")

    seen_ids = set()
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset record {position} must be a JSON object.")
        if not isinstance(record.get("id"), str) or not record["id"].strip():
            raise ValueError(f"Dataset record {position} has no valid id.")
        if record["id"] in seen_ids:
            raise ValueError(f"Dataset contains duplicate id: {record['id']}.")
        seen_ids.add(record["id"])
        if not isinstance(record.get("category"), str) or not record["category"].strip():
            raise ValueError(f"Dataset record {record['id']} has no valid category.")
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Dataset record {record['id']} has no valid text.")
        label = record.get("expected_label")
        if not isinstance(label, str) or label.upper() not in VALID_LABELS:
            raise ValueError(
                f"Dataset record {record['id']} has invalid expected_label: {label!r}."
            )
        record["expected_label"] = label.upper()
    return records


def classify_result(result: Any) -> tuple[str | None, str | None, str | None]:
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], bool)
        or not isinstance(result[1], str)
    ):
        return None, "error", f"check_safety returned an invalid result: {result!r}"

    is_safe, reason = result
    if is_safe:
        return "SAFE", "classified", reason
    if "was blocked by Llama Guard 2 as unsafe." in reason:
        return "UNSAFE", "classified", reason
    if "returned an unexpected response" in reason or "returned an empty response" in reason:
        return None, "unexpected_response", reason
    if "Could not run the " in reason and "safety check" in reason:
        return None, "error", reason
    return None, "error", f"Unrecognized check_safety result: {reason}"


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [
        record
        for record in records
        if record["actual_prediction"] in VALID_LABELS
    ]
    true_positive = sum(
        record["expected_label"] == "UNSAFE"
        and record["actual_prediction"] == "UNSAFE"
        for record in scored
    )
    false_positive = sum(
        record["expected_label"] == "SAFE"
        and record["actual_prediction"] == "UNSAFE"
        for record in scored
    )
    true_negative = sum(
        record["expected_label"] == "SAFE"
        and record["actual_prediction"] == "SAFE"
        for record in scored
    )
    false_negative = sum(
        record["expected_label"] == "UNSAFE"
        and record["actual_prediction"] == "SAFE"
        for record in scored
    )

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "case_count": len(records),
        "scored_case_count": len(scored),
        "accuracy": ratio(true_positive + true_negative, len(scored)),
        "precision": precision,
        "recall": recall,
        "f1_score": ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": ratio(
            false_positive, false_positive + true_negative
        ),
        "false_negative_rate": ratio(
            false_negative, false_negative + true_positive
        ),
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "confusion_matrix": {
            "true_positive_unsafe": true_positive,
            "false_positive_unsafe": false_positive,
            "true_negative_safe": true_negative,
            "false_negative_safe": false_negative,
        },
    }


def evaluate(
    records: list[dict[str, Any]],
    guard_type: str = "input",
    dataset_path: Path = DATASET_PATH,
) -> dict[str, Any]:
    if guard_type not in {"input", "output"}:
        raise ValueError("guard_type must be 'input' or 'output'.")
    if not records:
        raise ValueError("The Llama Guard dataset is empty.")
    results = []
    errors = []
    unexpected_responses = []

    for record in records:
        started = time.perf_counter()
        try:
            safety_result = llama_guard.check_safety(
                record["text"], guard_type=guard_type
            )
            prediction, status, reason = classify_result(safety_result)
        except (ConnectionError, RuntimeError, TimeoutError, ValueError) as error:
            prediction = None
            status = "error"
            reason = f"{type(error).__name__}: {error}"
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "id": record["id"],
            "category": record["category"],
            "text": record["text"],
            "expected_label": record["expected_label"],
            "actual_prediction": prediction,
            "status": status,
            "latency_ms": latency_ms,
            "reason": reason,
        }
        if "reason" in record:
            result["expected_reason"] = record["reason"]
        results.append(result)
        if status == "error":
            errors.append({"id": record["id"], "category": record["category"], "error": reason})
        elif status == "unexpected_response":
            unexpected_responses.append(
                {
                    "id": record["id"],
                    "category": record["category"],
                    "expected_label": record["expected_label"],
                    "response_detail": reason,
                }
            )

        print(
            f"{len(results)}/{len(records)} {record['id']}: "
            f"expected={record['expected_label']} actual={prediction or status} "
            f"latency={latency_ms:.1f} ms"
        )

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        categories[result["category"]].append(result)
    category_results = {}
    for category, category_records in sorted(categories.items()):
        category_results[category] = {
            **calculate_metrics(category_records),
            "average_latency_ms": sum(
                record["latency_ms"] for record in category_records
            )
            / len(category_records),
            "error_count": sum(record["status"] == "error" for record in category_records),
            "unexpected_response_count": sum(
                record["status"] == "unexpected_response" for record in category_records
            ),
        }

    return {
        "dataset": str(dataset_path),
        "guard_configuration": {
            "model": llama_guard.LLAMA_GUARD_MODEL,
            "url": llama_guard.LLAMA_GUARD_URL,
            "guard_type": guard_type,
        },
        "total_cases": len(results),
        "metrics": calculate_metrics(results),
        "average_latency_ms": sum(result["latency_ms"] for result in results)
        / len(results)
        if results
        else 0.0,
        "category_results": category_results,
        "error_count": len(errors),
        "errors": errors,
        "unexpected_response_count": len(unexpected_responses),
        "unexpected_responses": unexpected_responses,
        "results": results,
    }


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        f"# Llama Guard {report['guard_configuration']['guard_type'].title()} Safety Evaluation",
        "",
        f"- Cases: {report['total_cases']}",
        f"- Scored predictions: {metrics['scored_case_count']}",
        f"- Average latency: {report['average_latency_ms']:.2f} ms",
        f"- Errors: {report['error_count']}",
        f"- Unexpected responses: {report['unexpected_response_count']}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision (UNSAFE) | {metrics['precision']:.4f} |",
        f"| Recall (UNSAFE) | {metrics['recall']:.4f} |",
        f"| F1-score | {metrics['f1_score']:.4f} |",
        f"| False positives | {metrics['false_positives']} |",
        f"| False negatives | {metrics['false_negatives']} |",
        f"| False positive rate | {metrics['false_positive_rate']:.4f} |",
        f"| False negative rate | {metrics['false_negative_rate']:.4f} |",
        "",
        "Confusion counts use UNSAFE as the positive class: "
        f"TP={metrics['confusion_matrix']['true_positive_unsafe']}, "
        f"FP={metrics['confusion_matrix']['false_positive_unsafe']}, "
        f"TN={metrics['confusion_matrix']['true_negative_safe']}, "
        f"FN={metrics['confusion_matrix']['false_negative_safe']}.",
        "",
        "## Category-wise results",
        "",
        "| Category | Cases | Scored | Accuracy | Precision | Recall | F1 | FPR | FNR | Avg latency (ms) | Errors | Unexpected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, result in report["category_results"].items():
        lines.append(
            f"| {_markdown_cell(category)} | {result['case_count']} "
            f"| {result['scored_case_count']} | {result['accuracy']:.4f} "
            f"| {result['precision']:.4f} | {result['recall']:.4f} "
            f"| {result['f1_score']:.4f} | {result['false_positive_rate']:.4f} "
            f"| {result['false_negative_rate']:.4f} | {result['average_latency_ms']:.2f} "
            f"| {result['error_count']} | {result['unexpected_response_count']} |"
        )

    lines.extend(["", "## Per-case predictions", ""])
    lines.extend(
        [
            "| ID | Category | Expected | Actual | Status | Latency (ms) | Detail |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for result in report["results"]:
        lines.append(
            f"| {_markdown_cell(result['id'])} | {_markdown_cell(result['category'])} "
            f"| {result['expected_label']} | {result['actual_prediction'] or '—'} "
            f"| {result['status']} | {result['latency_ms']:.2f} "
            f"| {_markdown_cell(result['reason'] or '')} |"
        )

    lines.extend(["", "## Errors", ""])
    if report["errors"]:
        for error in report["errors"]:
            lines.append(
                f"- `{_markdown_cell(error['id'])}` "
                f"({_markdown_cell(error['category'])}): "
                f"{_markdown_cell(error['error'])}"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Unexpected model responses", ""])
    if report["unexpected_responses"]:
        for response in report["unexpected_responses"]:
            lines.append(
                f"- `{_markdown_cell(response['id'])}` "
                f"(expected {response['expected_label']}): "
                f"{_markdown_cell(response['response_detail'])}"
            )
    else:
        lines.append("None.")
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
        description="Evaluate Llama Guard input safety against labeled examples."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json-report", type=Path, default=JSON_REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_dataset(args.dataset)
    report = evaluate(records, guard_type="input", dataset_path=args.dataset.resolve())
    write_reports(report, args.json_report, args.markdown_report)
    print(f"\nEvaluation complete: {report['total_cases']} cases")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Recall: {report['metrics']['recall']:.4f}")
    print(f"F1-score: {report['metrics']['f1_score']:.4f}")
    print(f"False positive rate: {report['metrics']['false_positive_rate']:.4f}")
    print(f"False negative rate: {report['metrics']['false_negative_rate']:.4f}")
    print(f"Average latency: {report['average_latency_ms']:.2f} ms")
    print(f"Errors: {report['error_count']}")
    print(f"Unexpected responses: {report['unexpected_response_count']}")
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.markdown_report}")


if __name__ == "__main__":
    main()
