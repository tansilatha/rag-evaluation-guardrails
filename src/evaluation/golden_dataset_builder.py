import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from src.rag.rag_chat import OLLAMA_MODEL, OLLAMA_URL


PROJECT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_DIR / ".env")
DATA_DIR = PROJECT_DIR / "data"
SOURCE_PATH = DATA_DIR / "documents" / "machine_learning.txt"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
GOLDEN_DIR = PROJECT_DIR / "datasets" / "goldens"
RETRIEVAL_PATH = GOLDEN_DIR / "retrieval_goldens.json"
GENERATION_PATH = GOLDEN_DIR / "generation_goldens.json"
PIPELINE_PATH = GOLDEN_DIR / "pipeline_goldens.json"
CANDIDATES_PATH = GOLDEN_DIR / "rag_golden_candidates.jsonl"
REJECTED_PATH = GOLDEN_DIR / "rag_golden_rejected.jsonl"
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504, 529}
QUESTION_CATEGORIES = {
    "simple_factual",
    "conceptual",
    "multi_piece",
    "comparison",
    "application_based",
    "document_reasoning",
    "answer_absent",
}
CATEGORY_SEQUENCE = [
    "simple_factual",
    "conceptual",
    "multi_piece",
    "comparison",
    "application_based",
    "document_reasoning",
    "answer_absent",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def load_source_document() -> str:
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Source document not found: {SOURCE_PATH}")
    text = SOURCE_PATH.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Source document is empty: {SOURCE_PATH}")
    return text


def load_existing_chunks() -> list[dict[str, str]]:
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    chunks = []
    seen_ids = set()
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {CHUNKS_PATH} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or any(
                not isinstance(record.get(field), str) or not record[field].strip()
                for field in ("chunk_id", "source", "text")
            ):
                raise ValueError(
                    f"Chunk record at line {line_number} needs chunk_id, source, "
                    "and text strings."
                )
            chunk_id = record["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID in chunks file: {chunk_id}")
            seen_ids.add(chunk_id)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": record["source"],
                    "text": record["text"],
                }
            )

    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}")
    return chunks


def post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
    service_name: str,
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError(f"{service_name} returned a non-object JSON response.")
            return result
        except HTTPError as error:
            if error.code in RETRYABLE_STATUSES and attempt < 2:
                time.sleep(1.0 * (2**attempt))
                continue
            raise RuntimeError(
                f"{service_name} request failed with HTTP {error.code}."
            ) from error
        except (URLError, TimeoutError) as error:
            if attempt < 2:
                time.sleep(1.0 * (2**attempt))
                continue
            raise ConnectionError(
                f"Could not reach {service_name} after 3 attempts."
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError(f"{service_name} returned invalid JSON.") from error

    raise RuntimeError(f"{service_name} request failed after 3 attempts.")


def generate_candidate_questions(
    source_document: str,
    chunks: list[dict[str, str]],
    count: int,
) -> list[dict[str, Any]]:
    if count == 3:
        requested_categories = [
            "simple_factual",
            "comparison",
            "answer_absent",
        ]
    else:
        requested_categories = [
            CATEGORY_SEQUENCE[position % len(CATEGORY_SEQUENCE)]
            for position in range(count)
        ]
    chunk_text = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in chunks
    )
    category_counts = {
        category: requested_categories.count(category)
        for category in dict.fromkeys(requested_categories)
    }
    examples = []

    for category, category_count in category_counts.items():
        category_guidance = {
            "simple_factual": (
                "Ask what information each supervised training example contains."
            ),
            "conceptual": (
                "Ask for a plain-language explanation of a concept such as "
                "overfitting, underfitting, gradient descent, or a neural network."
            ),
            "multi_piece": (
                "Ask how training, validation, and test sets are used; the answer "
                "must combine their distinct roles."
            ),
            "comparison": (
                "Ask exactly how classification and regression differ in their "
                "outputs. The question must name both concepts and ask how they "
                "differ. The answer must state that classification predicts "
                "discrete categories and regression predicts continuous numerical "
                "values, with exact source quotations for both."
            ),
            "application_based": (
                "Ask how a documented concept or technique applies to a concrete "
                "scenario explicitly described in the document, such as grouping "
                "retail customers or predicting house prices."
            ),
            "document_reasoning": (
                "Ask why an overfitted model may perform poorly on unseen data, "
                "requiring the answer to connect learning noise or accidental "
                "patterns with generalization."
            ),
            "answer_absent": (
                "Ask for an exact model performance result, such as a test accuracy, "
                "which the full document does not provide. Mark answerable false, "
                "and use an empty answer and empty supporting_passages. Do not guess."
            ),
        }[category]
        prompt = f"""
Generate exactly {category_count} evaluation example(s) for the category
"{category}". {category_guidance}

Use only the source document and chunks below. Return only a JSON object with
an "examples" array. Each item must have:
- "question": a clear question that genuinely fits "{category}".
- "question_category": "{category}".
- "answerable": true for supported questions, false for answer_absent.
- "candidate_expected_answer": a concise supported answer, or an empty string
  for answer_absent.
- "supporting_passages": exact quotations copied from the document for
  answerable questions, or [] for answer_absent.

Never invent an answer, quotation, or ID. For answerable examples, copy a
complete short sentence exactly as evidence. For answer_absent, use the
question "What exact test accuracy did the model achieve?" if suitable.

SOURCE DOCUMENT:
{source_document}

CHUNKS:
{chunk_text}
""".strip()
        payload = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Generate candidate evaluation data grounded only in the "
                        "supplied document. Follow the requested category and "
                        "return valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        response = post_json(
            OLLAMA_URL,
            payload,
            {"Content-Type": "application/json"},
            timeout=180,
            service_name="Ollama",
        )
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("Ollama response did not contain generated JSON text.")
        try:
            result = json.loads(message["content"])
        except json.JSONDecodeError as error:
            raise ValueError("Ollama candidate response was not valid JSON.") from error
        category_examples = result.get("examples") if isinstance(result, dict) else None
        if not isinstance(category_examples, list):
            raise ValueError("Ollama response must contain an 'examples' array.")
        valid_examples = [
            example for example in category_examples if isinstance(example, dict)
        ]
        if len(valid_examples) < category_count:
            LOGGER.warning(
                "Ollama returned %d of %d requested %s example(s).",
                len(valid_examples),
                category_count,
                category,
            )
        for example in valid_examples[:category_count]:
            example["_requested_category"] = category
            example["_model_question_category"] = example.get("question_category")
            example["question_category"] = category
            examples.append(example)

    return examples


def extract_supporting_evidence(
    candidate: dict[str, Any],
    source_document: str,
    chunks: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    passages = candidate.get("supporting_passages")
    errors = []
    evidence = []
    if not isinstance(passages, list) or any(
        not isinstance(passage, str) or not passage.strip() for passage in passages
    ):
        return [], ["Supporting passages must be a list of non-empty strings."]

    source_normalized = normalize_text(source_document)
    for passage in passages:
        normalized_passage = normalize_text(passage)
        if normalized_passage not in source_normalized:
            errors.append("A proposed supporting passage is not in the source document.")
            continue

        matching_chunks = [
            chunk
            for chunk in chunks
            if normalized_passage in normalize_text(chunk["text"])
        ]
        if not matching_chunks:
            errors.append(
                "A source passage could not be traced to any actual document chunk."
            )
            continue
        for chunk in matching_chunks:
            item = {
                "passage": passage,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
            }
            if item not in evidence:
                evidence.append(item)
    return evidence, errors


def select_relevant_chunks(
    question: str, answer: str, chunks: list[dict[str, str]]
) -> list[dict[str, str]]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "the", "to", "what",
        "when", "where", "which", "with",
    }
    query_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", f"{question} {answer}".casefold())
        if token not in stop_words and len(token) > 1
    }
    if not query_terms:
        return []

    scored_chunks = []
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9]+", chunk["text"].casefold()))
        score = len(query_terms & chunk_terms)
        if score:
            scored_chunks.append((score, chunk))
    if not scored_chunks:
        return []

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "passage": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"],
        }
        for _, chunk in scored_chunks[:3]
    ]


def validate_chunk_ids(
    chunk_ids: list[str], chunks: list[dict[str, str]]
) -> list[str]:
    known_ids = {chunk["chunk_id"] for chunk in chunks}
    return [chunk_id for chunk_id in chunk_ids if chunk_id not in known_ids]


def validate_candidate(
    candidate: dict[str, Any],
    source_document: str,
    chunks: list[dict[str, str]],
) -> tuple[dict[str, Any], list[str]]:
    errors = []
    question = candidate.get("question")
    category = candidate.get("question_category")
    answerable = candidate.get("answerable")
    answer = candidate.get("candidate_expected_answer")

    if not isinstance(question, str) or not question.strip():
        errors.append("Question is missing or empty.")
    if category not in QUESTION_CATEGORIES:
        errors.append(f"Unsupported question category: {category!r}.")
    if not isinstance(answerable, bool):
        errors.append("Answerability must be a boolean.")
    if not isinstance(answer, str):
        errors.append("Candidate expected answer must be a string.")

    evidence, evidence_errors = extract_supporting_evidence(
        candidate, source_document, chunks
    )
    evidence_warnings = list(evidence_errors)

    if answerable is True:
        if category == "answer_absent":
            errors.append("An answer_absent example cannot be marked answerable.")
        if not isinstance(answer, str) or not answer.strip():
            errors.append("Answerable candidates need a non-empty expected answer.")
        if not evidence:
            evidence = select_relevant_chunks(
                question if isinstance(question, str) else "",
                answer if isinstance(answer, str) else "",
                chunks,
            )
        if not evidence:
            errors.append("Answerable candidates need traceable supporting evidence.")
    elif answerable is False:
        errors.extend(evidence_errors)
        if category != "answer_absent":
            errors.append("Unanswerable candidates must use the answer_absent category.")
        if isinstance(answer, str) and answer.strip():
            errors.append("Unanswerable candidates must not contain a factual answer.")
        if candidate.get("supporting_passages") != []:
            errors.append("Unanswerable candidates must not claim supporting passages.")
        evidence = []

    chunk_ids = sorted({item["chunk_id"] for item in evidence})
    missing_ids = validate_chunk_ids(chunk_ids, chunks)
    if missing_ids:
        errors.append("Unknown chunk ID(s): " + ", ".join(missing_ids))

    checked = dict(candidate)
    checked["generated_supporting_passages"] = candidate.get(
        "generated_supporting_passages",
        candidate.get("supporting_passages"),
    )
    checked["supporting_passages"] = [
        item["passage"] for item in evidence
    ]
    checked["supporting_evidence"] = evidence
    checked["relevant_chunk_ids"] = chunk_ids if answerable is True else []
    checked["source_document"] = SOURCE_PATH.name
    checked["expected_behavior"] = (
        "Answer using the trusted context."
        if answerable is True
        else "State that the available context does not contain enough information."
    )
    checked["validation_errors"] = errors
    checked["validation_warnings"] = evidence_warnings
    checked["chunk_id_validation_errors"] = [
        error
        for error in errors
        if "chunk ID" in error
        or "traced to any actual document chunk" in error
    ]
    return checked, errors


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    atomic_write_text(path, text)


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Record at {path}:{line_number} must be an object.")
            records.append(record)
    return records


def load_candidate_log() -> list[dict[str, Any]]:
    return load_jsonl_if_exists(CANDIDATES_PATH)


def save_candidate_logs(candidates: list[dict[str, Any]]) -> None:
    rejected = [
        candidate
        for candidate in candidates
        if candidate.get("review_status") == "rejected"
    ]
    write_jsonl(CANDIDATES_PATH, candidates)
    write_jsonl(REJECTED_PATH, rejected)
    load_jsonl_if_exists(CANDIDATES_PATH)
    load_jsonl_if_exists(REJECTED_PATH)


def save_golden_datasets(
    candidates: list[dict[str, Any]], overwrite: bool = False
) -> None:
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs must be unique before saving datasets.")

    accepted = [
        candidate
        for candidate in candidates
        if candidate.get("review_status") == "approved"
    ]
    if not accepted:
        raise ValueError("No human-approved candidates are available to build goldens.")
    output_paths = (RETRIEVAL_PATH, GENERATION_PATH, PIPELINE_PATH)
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not overwrite:
        paths = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"Refusing to overwrite existing golden dataset(s): {paths}. "
            "Pass --overwrite-goldens only after backing them up."
        )
    chunks = load_existing_chunks()
    chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
    known_chunk_ids = set(chunk_lookup)
    source_document = load_source_document()
    for candidate in accepted:
        if candidate.get("validation_errors"):
            raise ValueError(
                f"Approved candidate {candidate.get('id')} has validation errors."
            )
        relevant_ids = candidate.get("relevant_chunk_ids")
        evidence = candidate.get("supporting_evidence")
        answerable = candidate.get("answerable")
        expected_answer = candidate.get("candidate_expected_answer")
        expected_behavior = candidate.get("expected_behavior")
        if (
            not isinstance(candidate.get("question"), str)
            or not candidate["question"].strip()
            or not isinstance(relevant_ids, list)
            or any(
                not isinstance(chunk_id, str) or chunk_id not in known_chunk_ids
                for chunk_id in relevant_ids
            )
            or not isinstance(evidence, list)
            or not isinstance(answerable, bool)
        ):
            raise ValueError(
                f"Accepted candidate {candidate.get('id')} has invalid required fields."
            )
        evidence_ids = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError(
                    f"Accepted candidate {candidate.get('id')} has invalid evidence."
                )
            chunk_id = item.get("chunk_id")
            passage = item.get("passage")
            if (
                chunk_id not in chunk_lookup
                or not isinstance(passage, str)
                or not passage.strip()
                or normalize_text(passage) not in normalize_text(source_document)
                or normalize_text(passage)
                not in normalize_text(chunk_lookup[chunk_id]["text"])
            ):
                raise ValueError(
                    f"Accepted candidate {candidate.get('id')} has evidence "
                    "that cannot be traced to its source chunk."
                )
            evidence_ids.add(chunk_id)

        if answerable:
            if (
                not isinstance(expected_answer, str)
                or not expected_answer.strip()
                or expected_behavior != "Answer using the trusted context."
                or not evidence_ids
                or set(relevant_ids) != evidence_ids
            ):
                raise ValueError(
                    f"Accepted answerable candidate {candidate.get('id')} must "
                    "have a supported answer and matching evidence/chunk IDs."
                )
        elif (
            expected_answer not in ("", None)
            or expected_behavior
            != "State that the available context does not contain enough information."
            or relevant_ids
            or evidence
        ):
            raise ValueError(
                f"Accepted unanswerable candidate {candidate.get('id')} must "
                "record abstention behavior and have no relevant chunk IDs."
            )
    retrieval = []
    generation = []
    pipeline = []
    has_unanswerable = any(
        not candidate.get("answerable") for candidate in accepted
    )
    full_source_context = source_document if has_unanswerable else ""
    for candidate in accepted:
        ids = candidate["relevant_chunk_ids"]
        answerable = candidate["answerable"]
        common = {
            "id": candidate["id"],
            "question": candidate["question"],
            "source_document": candidate["source_document"],
            "question_category": candidate["question_category"],
            "review_status": "approved",
        }
        retrieval.append(
            {
                **common,
                "relevant_chunk_ids": ids,
                "supporting_evidence": candidate["supporting_evidence"],
            }
        )
        generation_context = (
            candidate["supporting_evidence"]
            if answerable
            else [{"source": SOURCE_PATH.name, "text": full_source_context}]
        )
        generation.append(
            {
                **common,
                "trusted_context": generation_context,
                "source_evidence": candidate["supporting_evidence"],
                "expected_answer": candidate["candidate_expected_answer"]
                if answerable
                else None,
                "expected_behavior": candidate["expected_behavior"],
            }
        )
        pipeline.append(
            {
                "id": candidate["id"],
                "question": candidate["question"],
                "trusted_evidence": candidate["supporting_evidence"],
                "expected_answer": candidate["candidate_expected_answer"]
                if answerable
                else None,
                "relevant_chunk_ids": ids,
                "expected_behavior": candidate["expected_behavior"],
                "question_category": candidate["question_category"],
                "review_status": "approved",
            }
        )

    write_json(RETRIEVAL_PATH, retrieval)
    write_json(GENERATION_PATH, generation)
    write_json(PIPELINE_PATH, pipeline)

    for path in output_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON was written to {path}: {error}") from error
        if not isinstance(value, list):
            raise ValueError(f"Dataset output must be a JSON array: {path}")


def create_candidate_record(
    raw_candidate: dict[str, Any], candidate_id: str
) -> dict[str, Any]:
    return {
        "id": candidate_id,
        "question": raw_candidate.get("question"),
        "question_category": raw_candidate.get("question_category"),
        "model_question_category": raw_candidate.get("_model_question_category"),
        "answerable": raw_candidate.get("answerable"),
        "candidate_expected_answer": raw_candidate.get("candidate_expected_answer"),
        "supporting_passages": raw_candidate.get("supporting_passages"),
        "review_status": "pending",
        "review_notes": "Candidate requires human review before use as a golden.",
    }


def next_candidate_id(candidates: list[dict[str, Any]]) -> str:
    largest = 0
    for candidate in candidates:
        match = re.fullmatch(r"golden_(\d+)", str(candidate.get("id", "")))
        if match:
            largest = max(largest, int(match.group(1)))
    return f"golden_{largest + 1:04d}"


def print_test_candidate(candidate: dict[str, Any]) -> None:
    print(f"\nID: {candidate['id']}")
    print(f"Generated question: {candidate.get('question')}")
    print(
        "Candidate expected answer: "
        f"{candidate.get('candidate_expected_answer') or '(none; abstain)'}"
    )
    print(f"Supporting evidence: {candidate.get('supporting_evidence', [])}")
    print(f"Relevant chunk IDs: {candidate.get('relevant_chunk_ids', [])}")
    print(f"Review status: {candidate.get('review_status')}")
    print(f"Review notes: {candidate.get('review_notes')}")
    if candidate.get("validation_errors"):
        print(f"Validation issues: {candidate['validation_errors']}")
    if candidate.get("validation_warnings"):
        print(f"Evidence selection notes: {candidate['validation_warnings']}")


def print_report(
    candidates: list[dict[str, Any]],
    generated_this_run: int,
) -> None:
    statuses = [candidate.get("review_status") for candidate in candidates]
    approved = statuses.count("approved")
    rejected = statuses.count("rejected")
    pending = statuses.count("pending")
    validation_failures = sum(
        bool(candidate.get("validation_errors")) for candidate in candidates
    )
    chunk_id_failures = sum(
        bool(candidate.get("chunk_id_validation_errors"))
        for candidate in candidates
    )
    print("\nGolden dataset report")
    print(f"New candidates generated this run: {generated_this_run}")
    print(f"Candidates saved in total: {len(candidates)}")
    category_counts = {}
    for candidate in candidates:
        category = candidate.get("question_category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    print(f"Saved question categories: {category_counts}")
    print(f"Approved by human review: {approved}")
    print(f"Rejected: {rejected}")
    print(f"Pending human review: {pending}")
    print(f"Failing chunk-ID validation: {chunk_id_failures}")
    print(f"Failing candidate/evidence validation: {validation_failures}")
    print(f"Candidate log: {CANDIDATES_PATH}")


def run_candidate_generation(args: argparse.Namespace) -> None:
    source = load_source_document()
    chunks = load_existing_chunks()
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_log()
    existing_questions = {
        normalize_text(str(candidate.get("question", "")))
        for candidate in candidates
        if candidate.get("question")
    }
    count = 3 if args.test_mode else args.count

    raw_candidates = generate_candidate_questions(source, chunks, count)
    generated = 0
    for raw_candidate in raw_candidates:
        if generated >= count:
            break
        question = raw_candidate.get("question")
        normalized_question = normalize_text(question) if isinstance(question, str) else ""
        if normalized_question and normalized_question in existing_questions:
            LOGGER.info("Skipping duplicate generated question.")
            continue

        record = create_candidate_record(raw_candidate, next_candidate_id(candidates))
        checked, errors = validate_candidate(record, source, chunks)
        record.update(checked)
        record["id"] = next_candidate_id(candidates)
        if errors:
            record["review_status"] = "rejected"
            record["review_notes"] = "Source evidence validation failed."
        else:
            record["review_status"] = "pending"
            record["review_notes"] = (
                "Check answer correctness, category, and evidence before approval."
            )
        candidates.append(record)
        existing_questions.add(normalized_question)
        generated += 1
        if args.test_mode:
            print_test_candidate(record)

    save_candidate_logs(candidates)
    print_report(candidates, generated)
    if generated == 0:
        LOGGER.warning("No new candidates were generated.")


def build_reviewed_goldens(overwrite: bool) -> None:
    candidates = load_candidate_log()
    save_golden_datasets(candidates, overwrite=overwrite)
    print(f"Reviewed RAG goldens saved under: {GOLDEN_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate candidate data and build human-reviewed RAG goldens."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="candidate batch size (default: 10; ignored in test mode)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="generate at most 3 examples and display each result",
    )
    parser.add_argument(
        "--build-reviewed",
        action="store_true",
        help="build RAG goldens from candidates manually marked approved",
    )
    parser.add_argument(
        "--overwrite-goldens",
        action="store_true",
        help="allow replacing existing golden files (back them up first)",
    )
    args = parser.parse_args()
    if args.count < 1 or args.count > 50:
        parser.error("--count must be between 1 and 50")
    if args.overwrite_goldens and not args.build_reviewed:
        parser.error("--overwrite-goldens requires --build-reviewed")
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.build_reviewed:
            build_reviewed_goldens(args.overwrite_goldens)
        else:
            run_candidate_generation(args)
    except (
        ConnectionError,
        FileExistsError,
        FileNotFoundError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        LOGGER.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
