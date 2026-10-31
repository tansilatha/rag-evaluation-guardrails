import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.rag import rag_chat
from sentence_transformers import SentenceTransformer

from src.rag import retrieve


PROJECT_DIR = Path(__file__).resolve().parents[2]
GOLDEN_PATH = PROJECT_DIR / "datasets" / "goldens" / "retrieval_goldens.json"
REPORT_PATH = PROJECT_DIR / "reports" / "context_evaluation.json"


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b'{"message": {"content": "captured"}}'


def load_answerable_goldens(path: Path = GOLDEN_PATH) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Retrieval goldens must be a JSON array.")
    answerable = []
    for record in records:
        if not isinstance(record, dict):
            continue
        relevant_ids = record.get("relevant_chunk_ids")
        question = record.get("question")
        if (
            isinstance(question, str)
            and question.strip()
            and isinstance(relevant_ids, list)
            and relevant_ids
        ):
            answerable.append(record)
    return answerable


def inspect_generator_payload(
    question: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    captured = {}

    def capture(request, timeout):
        captured["request"] = request
        return _FakeResponse()

    with patch.object(rag_chat, "urlopen", side_effect=capture):
        rag_chat.generate_answer(question, chunks)

    payload = json.loads(captured["request"].data.decode("utf-8"))
    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("RAG generator request did not contain system and user messages.")
    user_content = messages[1].get("content")
    if not isinstance(user_content, str):
        raise ValueError("RAG generator user message did not contain text content.")
    return {
        "payload": payload,
        "user_content": user_content,
    }


def evaluate(report_path: Path = REPORT_PATH) -> dict[str, Any]:
    records = load_answerable_goldens()
    index, metadata = retrieve.load_index_and_metadata()
    model = SentenceTransformer(retrieve.MODEL_NAME)
    results = []
    duplicate_cases = 0
    empty_chunk_cases = 0
    omission_cases = 0

    for record in records:
        chunks = rag_chat.retrieve_chunks(record["question"], model, index, metadata)
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            duplicate_cases += 1
        if any(not isinstance(chunk.get("text"), str) or not chunk["text"] for chunk in chunks):
            empty_chunk_cases += 1
        captured = inspect_generator_payload(record["question"], chunks)
        user_content = captured["user_content"]
        missing_chunks = [
            chunk["chunk_id"]
            for chunk in chunks
            if f"[{chunk['chunk_id']}]\n{chunk['text']}" not in user_content
        ]
        if missing_chunks:
            omission_cases += 1
        results.append(
            {
                "id": record.get("id"),
                "question": record["question"],
                "retrieved_chunk_ids": chunk_ids,
                "retrieved_chunk_count": len(chunks),
                "context_characters": len(user_content),
                "missing_chunk_ids_from_generator_context": missing_chunks,
                "question_in_generator_payload": f"Question: {record['question']}" in user_content,
                "generator_model": captured["payload"].get("model"),
            }
        )

    context_lengths = [result["context_characters"] for result in results]
    report = {
        "golden_records_available": len(records),
        "evaluated_questions": len(results),
        "retriever_configuration": {
            "model": retrieve.MODEL_NAME,
            "index_vectors": int(index.ntotal),
            "index_dimensions": int(index.d),
            "top_k": rag_chat.TOP_K,
        },
        "context_assembly_configuration": {
            "implementation": "rag_chat.generate_answer",
            "format": "[chunk_id]\\nchunk_text",
            "chunk_separator": "\\n\\n",
            "configured_context_length_limit": None,
        },
        "checks": {
            "all_retrieved_chunks_reach_generator": omission_cases == 0,
            "empty_retrieved_chunks": empty_chunk_cases,
            "duplicate_retrieved_chunk_cases": duplicate_cases,
            "context_omission_cases": omission_cases,
            "questions_reach_generator": all(
                result["question_in_generator_payload"] for result in results
            ),
            "maximum_observed_context_characters": max(context_lengths, default=0),
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluated questions: {report['evaluated_questions']}")
    print(f"All chunks reach generator: {report['checks']['all_retrieved_chunks_reach_generator']}")
    print(f"Empty retrieved chunks: {report['checks']['empty_retrieved_chunks']}")
    print(f"Duplicate retrieved-chunk cases: {report['checks']['duplicate_retrieved_chunk_cases']}")
    print(f"Context omission cases: {report['checks']['context_omission_cases']}")
    print(f"Maximum observed context characters: {report['checks']['maximum_observed_context_characters']}")
    print(f"Report saved to: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate existing RAG context assembly.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    evaluate(args.report)


if __name__ == "__main__":
    main()
