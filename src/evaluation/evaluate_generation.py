import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from openrouter import OpenRouter
from src.rag import rag_chat
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env", override=True)
GOLDEN_PATH = PROJECT_DIR / "datasets" / "goldens" / "generation_goldens.json"
REPORT_PATH = PROJECT_DIR / "reports" / "generation_evaluation.json"
JUDGE_TIMEOUT_SECONDS = 120
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "openrouter").casefold()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "baseten/mammoth-1:free")
# Backward-compatible name for existing OpenRouter-focused tests and callers.
OPENROUTER_MODEL = JUDGE_MODEL
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
AION_API_KEY_ENV = "AION_API_KEY"
AION_BASE_URL_ENV = "AION_BASE_URL"
JUDGE_MAX_TOKENS = 8192
JUDGE_RESPONSE_RETRY_LIMIT = 1
METRIC_NAMES = ("correctness", "faithfulness", "relevance")


def load_generation_goldens(
    path: Path = GOLDEN_PATH,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Generation golden dataset not found: {path}")
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Generation golden dataset contains invalid JSON: {error}") from error
    if not isinstance(records, list):
        raise ValueError("Generation golden dataset must be a JSON array.")
    return records


def is_unanswerable(record: dict[str, Any]) -> bool:
    return record.get("expected_answer") is None


def prepare_generation_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    prepared = []
    skipped = []
    for position, record in enumerate(records, start=1):
        record_id = (
            str(record.get("id", f"record_{position}"))
            if isinstance(record, dict)
            else f"record_{position}"
        )
        if not isinstance(record, dict):
            skipped.append({"id": record_id, "reason": "Record is not a JSON object."})
            continue
        question = record.get("question")
        if not isinstance(question, str) or not question.strip():
            skipped.append({"id": record_id, "reason": "Question is missing or empty."})
            continue
        expected_answer = record.get("expected_answer")
        if expected_answer is not None and (
            not isinstance(expected_answer, str) or not expected_answer.strip()
        ):
            skipped.append(
                {"id": record_id, "reason": "expected_answer must be a non-empty string or null."}
            )
            continue
        expected_behavior = record.get("expected_behavior")
        if not isinstance(expected_behavior, str) or not expected_behavior.strip():
            skipped.append(
                {"id": record_id, "reason": "expected_behavior is missing or empty."}
            )
            continue
        trusted_context = record.get("trusted_context")
        if not isinstance(trusted_context, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            for item in trusted_context
        ):
            skipped.append(
                {"id": record_id, "reason": "trusted_context must contain text records."}
            )
            continue
        if expected_answer is None and record.get("question_category") != "answer_absent":
            skipped.append(
                {"id": record_id, "reason": "Null expected_answer is not marked answer_absent."}
            )
            continue
        prepared.append(
            {
                "id": record_id,
                "question": question.strip(),
                "expected_answer": expected_answer.strip() if isinstance(expected_answer, str) else None,
                "expected_behavior": expected_behavior.strip(),
                "trusted_context": trusted_context,
                "question_category": record.get("question_category"),
            }
        )
    return prepared, skipped


def validate_judge_result(result: Any, unanswerable: bool) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Judge response must be a JSON object.")
    validated = {}
    for metric in METRIC_NAMES:
        value = result.get(f"{metric}_score")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Judge response has an invalid {metric}_score.")
        validated[f"{metric}_score"] = float(value)
    explanation = result.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("Judge response must contain a non-empty explanation.")
    unsupported_claims = result.get("unsupported_claims")
    if not isinstance(unsupported_claims, list) or any(
        not isinstance(claim, str) for claim in unsupported_claims
    ):
        raise ValueError("Judge response unsupported_claims must be a string list.")
    abstention_correct = result.get("abstention_correct")
    if unanswerable and not isinstance(abstention_correct, bool):
        raise ValueError("Judge response must contain boolean abstention_correct for an unanswerable question.")
    if not unanswerable and abstention_correct is not None and not isinstance(abstention_correct, bool):
        raise ValueError("Judge response abstention_correct must be boolean or null.")
    validated["explanation"] = explanation.strip()
    validated["unsupported_claims"] = unsupported_claims
    validated["abstention_correct"] = abstention_correct
    return validated


def judge_prompt(
    question: str,
    generated_answer: str,
    retrieved_context: list[dict[str, Any]],
    expected_answer: str | None,
    expected_behavior: str,
) -> str:
    context = "\n\n".join(
        f"[{item.get('chunk_id', 'unknown')}]\n{item.get('text', '')}"
        for item in retrieved_context
    )
    reference = expected_answer if expected_answer is not None else "No reference answer; this question is expected to be unanswerable."
    return f"""Evaluate the generated answer using only the supplied reference and retrieved context.
Return only one compact, valid JSON object with exactly these fields. Do not include markdown or reasoning in the JSON. Keep the explanation under 40 words and each unsupported claim description concise:
- correctness_score: number from 0 to 1
- faithfulness_score: number from 0 to 1
- relevance_score: number from 0 to 1
- explanation: short explanation
- unsupported_claims: array of unsupported claim strings, or []
- abstention_correct: boolean for an unanswerable question, otherwise null

Correctness compares the answer with the reference information. Faithfulness checks support in retrieved context. Relevance checks whether it answers the question. For an unanswerable question, score correctness based on whether it follows the expected abstention behavior; do not penalize it for lacking a reference answer.

QUESTION:
{question}

EXPECTED BEHAVIOR:
{expected_behavior}

REFERENCE ANSWER:
{reference}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{generated_answer}""".strip()


def request_openrouter_judge(prompt: str) -> dict[str, Any]:
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV, "").strip()
    if not api_key:
        raise PermissionError(
            f"{OPENROUTER_API_KEY_ENV} is not set; cannot call OpenRouter."
        )

    try:
        with OpenRouter(
            api_key=api_key,
            timeout_ms=JUDGE_TIMEOUT_SECONDS * 1000,
        ) as client:
            response = client.chat.send(
                model=JUDGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict RAG evaluation judge. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                max_tokens=JUDGE_MAX_TOKENS,
            )
    except Exception as error:
        message = str(error)
        if "401" in message or "authentication" in message.casefold():
            raise RuntimeError(f"OpenRouter authentication failed: {message}") from error
        if "429" in message or "rate limit" in message.casefold():
            raise RuntimeError(f"OpenRouter rate limit reached: {message}") from error
        raise RuntimeError(f"OpenRouter judge request failed: {message}") from error

    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenRouter judge response did not contain choices.")
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason in {"length", "incomplete"}:
        raise ValueError(
            f"OpenRouter judge generation was incomplete (finish_reason={finish_reason})."
        )
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenRouter judge response contained empty message content.")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("OpenRouter judge content was not valid JSON.") from error
    return result


def request_gemini_judge(prompt: str) -> dict[str, Any]:
    api_key = os.environ.get(GEMINI_API_KEY_ENV, "").strip()
    if not api_key:
        raise PermissionError(
            f"{GEMINI_API_KEY_ENV} is not set; cannot call Gemini."
        )
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "max_output_tokens": JUDGE_MAX_TOKENS,
            },
        )
    except Exception as error:
        message = str(error)
        if "401" in message or "authentication" in message.casefold() or "api key" in message.casefold():
            raise RuntimeError(f"Gemini authentication failed: {message}") from error
        if "429" in message or "quota" in message.casefold() or "rate limit" in message.casefold():
            raise RuntimeError(f"Gemini rate limit or quota reached: {message}") from error
        raise RuntimeError(f"Gemini judge request failed: {message}") from error

    content = getattr(response, "text", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Gemini judge response contained empty text.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Gemini judge content was not valid JSON.") from error


def request_aion_judge(prompt: str) -> dict[str, Any]:
    api_key = os.environ.get(AION_API_KEY_ENV, "").strip()
    base_url = os.environ.get(AION_BASE_URL_ENV, "").strip()
    if not api_key:
        raise PermissionError(f"{AION_API_KEY_ENV} is not set; cannot call Aion.")
    if not base_url:
        raise ValueError(f"{AION_BASE_URL_ENV} is not set; cannot call Aion.")
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=JUDGE_TIMEOUT_SECONDS,
        )
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict RAG evaluation judge. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=JUDGE_MAX_TOKENS,
        )
    except Exception as error:
        message = str(error)
        if "401" in message or "authentication" in message.casefold() or "api key" in message.casefold():
            raise RuntimeError(f"Aion authentication failed: {message}") from error
        if "429" in message or "quota" in message.casefold() or "rate limit" in message.casefold():
            raise RuntimeError(f"Aion rate limit or quota reached: {message}") from error
        raise RuntimeError(f"Aion judge request failed: {message}") from error

    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        raise ValueError("Aion judge response did not contain choices.")
    choice = choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason in {"length", "incomplete"}:
        raise ValueError(f"Aion judge generation was incomplete (finish_reason={finish_reason}).")
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Aion judge response contained empty message content.")
    try:
        return json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("Aion judge content was not valid JSON.") from error


def request_judge(prompt: str) -> dict[str, Any]:
    if JUDGE_PROVIDER == "aion":
        return request_aion_judge(prompt)
    if JUDGE_PROVIDER == "gemini":
        return request_gemini_judge(prompt)
    if JUDGE_PROVIDER == "openrouter":
        return request_openrouter_judge(prompt)
    raise RuntimeError(
        f"Unsupported JUDGE_PROVIDER: {JUDGE_PROVIDER}. "
        "Expected aion, gemini, or openrouter."
    )


def request_and_validate_judge(
    prompt: str,
    unanswerable: bool,
) -> dict[str, Any]:
    last_error = None
    for attempt in range(JUDGE_RESPONSE_RETRY_LIMIT + 1):
        try:
            response = request_judge(prompt)
            return validate_judge_result(response, unanswerable)
        except ValueError as error:
            last_error = error
            if attempt >= JUDGE_RESPONSE_RETRY_LIMIT:
                raise
    raise ValueError(f"Judge response validation failed: {last_error}")


def calculate_aggregate_scores(results: list[dict[str, Any]]) -> dict[str, float]:
    scored_results = [result for result in results if isinstance(result.get("judge"), dict)]
    if not scored_results:
        return {metric: 0.0 for metric in METRIC_NAMES}
    return {
        metric: statistics.mean(
            float(result["judge"][f"{metric}_score"]) for result in scored_results
        )
        for metric in METRIC_NAMES
    }


def evaluate(
    golden_path: Path = GOLDEN_PATH,
    report_path: Path = REPORT_PATH,
) -> dict[str, Any]:
    raw_records = load_generation_goldens(golden_path)
    records, skipped = prepare_generation_records(raw_records)
    index, metadata = rag_chat.load_index_and_metadata()
    model = SentenceTransformer(rag_chat.MODEL_NAME)
    results = []
    failures = []
    infrastructure_errors = []
    generation_latencies = []
    judge_latencies = []

    for item in records:
        generation_started = time.perf_counter()
        try:
            chunks = rag_chat.retrieve_chunks(item["question"], model, index, metadata)
            generated_answer = rag_chat.generate_answer(item["question"], chunks)
        except Exception as error:
            failure = {"id": item["id"], "stage": "generation", "error": str(error)}
            failures.append(failure)
            infrastructure_errors.append(failure)
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "expected_behavior": item["expected_behavior"],
                    "generated_answer": None,
                    "retrieved_context": [],
                    "evaluation_status": "infrastructure_error",
                    "error": str(error),
                }
            )
            continue
        generation_latency = (time.perf_counter() - generation_started) * 1000
        generation_latencies.append(generation_latency)

        judge_started = time.perf_counter()
        try:
            raw_judge = request_judge(
                judge_prompt(
                    item["question"],
                    generated_answer,
                    chunks,
                    item["expected_answer"],
                    item["expected_behavior"],
                )
            )
            judge = validate_judge_result(raw_judge, is_unanswerable(item))
        except Exception as error:
            failure = {"id": item["id"], "stage": "judge", "error": str(error)}
            failures.append(failure)
            infrastructure_errors.append(failure)
            results.append(
                {
                    "id": item["id"],
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "expected_behavior": item["expected_behavior"],
                    "generated_answer": generated_answer,
                    "retrieved_context": chunks,
                    "generation_latency_ms": generation_latency,
                    "evaluation_status": "infrastructure_error",
                    "error": str(error),
                }
            )
            continue
        judge_latency = (time.perf_counter() - judge_started) * 1000
        judge_latencies.append(judge_latency)
        result = {
            "id": item["id"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "expected_behavior": item["expected_behavior"],
            "generated_answer": generated_answer,
            "retrieved_context": chunks,
            "generation_latency_ms": generation_latency,
            "judge_latency_ms": judge_latency,
            "judge": judge,
            "evaluation_status": "evaluated",
        }
        if is_unanswerable(item) and judge["abstention_correct"] is False:
            result["evaluation_status"] = "failed"
            result["failure_reason"] = "Generated answer did not correctly abstain."
            failures.append(
                {"id": item["id"], "stage": "evaluation", "error": result["failure_reason"]}
            )
        results.append(result)

    report = {
        "golden_records": len(raw_records),
        "evaluated_questions": len(results),
        "scored_question_count": sum(
            1 for result in results if isinstance(result.get("judge"), dict)
        ),
        "skipped_question_count": len(skipped),
        "failed_question_count": len(failures),
        "skipped_questions": skipped,
        "failures": failures,
        "infrastructure_errors": infrastructure_errors,
        "evaluation_failures": [
            failure for failure in failures if failure not in infrastructure_errors
        ],
        "aggregate_scores": calculate_aggregate_scores(results),
        "average_generation_latency_ms": statistics.mean(generation_latencies) if generation_latencies else 0.0,
        "average_judge_latency_ms": statistics.mean(judge_latencies) if judge_latencies else 0.0,
        "generator_configuration": {
            "module": "rag_chat",
            "model": rag_chat.OLLAMA_MODEL,
            "url": rag_chat.OLLAMA_URL,
            "retrieval_top_k": rag_chat.TOP_K,
            "embedding_model": rag_chat.MODEL_NAME,
        },
        "judge_configuration": {
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "api_key_environment_variable": (
                GEMINI_API_KEY_ENV
                if JUDGE_PROVIDER == "gemini"
                else AION_API_KEY_ENV
                if JUDGE_PROVIDER == "aion"
                else OPENROUTER_API_KEY_ENV
            ),
            "max_tokens": JUDGE_MAX_TOKENS,
            "structured_output": True,
            "score_semantics": "Provider score fields validated by validate_judge_result.",
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(report, report_path)
    return report


def print_report(report: dict[str, Any], report_path: Path) -> None:
    print(f"Golden records: {report['golden_records']}")
    print(f"Evaluated: {report['evaluated_questions']}")
    print(f"Skipped: {report['skipped_question_count']}")
    print(f"Failed: {report['failed_question_count']}")
    for metric, score in report["aggregate_scores"].items():
        print(f"{metric.capitalize()}: {score:.3f}")
    print(f"Average generation latency: {report['average_generation_latency_ms']:.2f} ms")
    print(f"Average judge latency: {report['average_judge_latency_ms']:.2f} ms")
    for error in report["infrastructure_errors"]:
        print(f"{error['stage']} error for {error['id']}: {error['error']}")
    print(f"Report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated RAG answers against generation goldens.")
    parser.add_argument("--goldens", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        evaluate(args.goldens, args.report)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Generation evaluation error: {error}") from error


if __name__ == "__main__":
    main()
