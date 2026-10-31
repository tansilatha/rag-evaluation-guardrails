import argparse
from pathlib import Path

from src.evaluation import evaluate_llama_guard


DATASET_PATH = evaluate_llama_guard.OUTPUT_DATASET_PATH
JSON_REPORT_PATH = evaluate_llama_guard.REPORT_DIR / "output_safety_evaluation.json"
MARKDOWN_REPORT_PATH = evaluate_llama_guard.REPORT_DIR / "output_safety_evaluation.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Llama Guard output safety against labeled examples."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json-report", type=Path, default=JSON_REPORT_PATH)
    parser.add_argument("--markdown-report", type=Path, default=MARKDOWN_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = evaluate_llama_guard.load_dataset(args.dataset)
    report = evaluate_llama_guard.evaluate(
        records, guard_type="output", dataset_path=args.dataset.resolve()
    )
    evaluate_llama_guard.write_reports(
        report, args.json_report, args.markdown_report
    )
    print(f"\nEvaluation complete: {report['total_cases']} cases")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}")
    print(f"Precision: {report['metrics']['precision']:.4f}")
    print(f"Recall: {report['metrics']['recall']:.4f}")
    print(f"F1-score: {report['metrics']['f1_score']:.4f}")
    print(f"Average latency: {report['average_latency_ms']:.2f} ms")
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.markdown_report}")


if __name__ == "__main__":
    main()
