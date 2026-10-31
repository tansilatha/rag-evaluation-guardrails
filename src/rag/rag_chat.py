import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sentence_transformers import SentenceTransformer

from src.guardrails.llama_guard import (
    check_safety,
)
from src.rag.retrieve import MODEL_NAME, embed_question, load_index_and_metadata


OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"
TOP_K = 3

SYSTEM_PROMPT = (
    "Treat the retrieved context as the primary source for answering the user's "
    "question. Answer the exact question asked and do not go off-topic. Explain "
    "concepts in simple, clear language suitable for a beginner. Use only "
    "information supported by the retrieved context for factual claims about "
    "the source document. Do not invent facts, definitions, examples, statistics, "
    "or explanations that the context does not support. If the context does not "
    "contain enough information to answer, explicitly say that the provided "
    "context does not contain enough information. If the context supports only "
    "part of the question, answer the supported part and clearly identify what "
    "is missing. For comparison questions, explain the differences clearly, using "
    "bullet points when helpful. Keep answers concise by default, but include "
    "enough detail to answer completely. Before responding, check that your answer "
    "does not contradict the retrieved context."
)


def retrieve_chunks(question, model, index, metadata):
    query = embed_question(model, question)
    scores, positions = index.search(query, TOP_K)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or position >= len(metadata):
            raise ValueError(f"FAISS returned invalid vector position {int(position)}.")

        record = metadata[int(position)]
        results.append(
            {
                "chunk_id": record["chunk_id"],
                "text": record["text"],
                "score": float(score),
            }
        )
    return results


def generate_answer(question, chunks):
    context = "\n\n".join(
        f"[{chunk['chunk_id']}]\n{chunk['text']}" for chunk in chunks
    )
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Retrieved context:\n{context}\n\nQuestion: {question}",
            },
        ],
    }
    request = Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {error.code}: {details}"
        ) from error
    except URLError as error:
        raise ConnectionError(
            "Could not connect to Ollama. Make sure Ollama is running at "
            f"{OLLAMA_URL}."
        ) from error
    except TimeoutError as error:
        raise ConnectionError("The request to Ollama timed out.") from error

    if not isinstance(result, dict):
        raise ValueError("Ollama returned an unexpected response.")
    message = result.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Ollama response did not contain a message with answer text.")
    return message["content"]


def run_guarded_rag(question, model, index, metadata) -> dict[str, Any]:
    input_safe, input_reason = check_safety(question, guard_type="input")
    result: dict[str, Any] = {
        "status": "input_blocked",
        "input_safe": input_safe,
        "input_reason": input_reason,
        "retrieved_chunks": [],
        "output_safe": None,
        "output_reason": None,
    }
    if not input_safe:
        return result

    try:
        chunks = retrieve_chunks(question, model, index, metadata)
        answer = generate_answer(question, chunks)
    except (ConnectionError, RuntimeError, ValueError) as error:
        result["status"] = "error"
        result["error"] = str(error)
        return result

    result["retrieved_chunks"] = chunks
    output_safe, output_reason = check_safety(answer, guard_type="output")
    result["output_safe"] = output_safe
    result["output_reason"] = output_reason
    result["status"] = "completed" if output_safe else "output_blocked"
    if output_safe:
        result["answer"] = answer
    return result


def main():
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer(MODEL_NAME)
    print("RAG chat ready. Type 'exit' to quit.")

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except EOFError:
            print("\nExiting.")
            break

        if question.lower() == "exit":
            print("Exiting.")
            break
        if not question:
            print("Please enter a non-empty question.")
            continue

        try:
            result = run_guarded_rag(question, model, index, metadata)
            if result["status"] == "error":
                print(f"\nError: {result['error']}")
                continue
            input_safe = result["input_safe"]
            print(f"Input safety: {'SAFE' if input_safe else 'UNSAFE'}")
            if result["status"] == "input_blocked":
                print(f"Safety block (input): {result['input_reason']}")
                continue

            print("\nRetrieved chunks:")
            for chunk in result["retrieved_chunks"]:
                print(f"- {chunk['chunk_id']} (similarity: {chunk['score']:.4f})")

            output_safe = result["output_safe"]
            print(f"Output safety: {'SAFE' if output_safe else 'UNSAFE'}")
            if not output_safe:
                print(f"Safety block (output): {result['output_reason']}")
                continue

            print(f"\nAnswer:\n{result['answer']}")
        except (ConnectionError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()
