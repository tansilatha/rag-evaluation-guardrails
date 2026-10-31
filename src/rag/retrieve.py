import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "faiss_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_VECTOR_COUNT = 19
EXPECTED_DIMENSIONS = 384
DEFAULT_TOP_K = 5


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("top-k must be a positive integer") from error

    if parsed < 1:
        raise argparse.ArgumentTypeError("top-k must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve the most similar chunks for each question."
    )
    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=DEFAULT_TOP_K,
        help=f"number of chunks to show (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()
    if args.top_k > EXPECTED_VECTOR_COUNT:
        parser.error(f"--top-k cannot exceed {EXPECTED_VECTOR_COUNT} stored vectors")
    return args


def load_index_and_metadata() -> tuple[faiss.Index, list[dict[str, object]]]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"FAISS metadata not found: {METADATA_PATH}")

    index = faiss.read_index(str(INDEX_PATH))
    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError("FAISS metadata must be a JSON array.")
    if index.ntotal != EXPECTED_VECTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_COUNT} indexed vectors, got {index.ntotal}."
        )
    if len(metadata) != EXPECTED_VECTOR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VECTOR_COUNT} metadata records, got {len(metadata)}."
        )
    if index.d != EXPECTED_DIMENSIONS:
        raise ValueError(
            f"Expected index dimension {EXPECTED_DIMENSIONS}, got {index.d}."
        )
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError(
            "The FAISS index must use inner product over normalized vectors "
            "for cosine similarity."
        )

    indexed_vectors = index.reconstruct_n(0, index.ntotal)
    if not np.isfinite(indexed_vectors).all():
        raise ValueError("FAISS index contains non-finite vectors.")
    if not np.allclose(np.linalg.norm(indexed_vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("FAISS index vectors are not normalized for cosine search.")

    required_fields = ("chunk_id", "source", "text")
    for position, record in enumerate(metadata):
        if not isinstance(record, dict) or record.get("vector_id") != position:
            raise ValueError(
                f"Metadata record {position} does not map to vector position "
                f"{position}."
            )
        if any(not isinstance(record.get(field), str) for field in required_fields):
            raise ValueError(
                f"Metadata record {position} is missing chunk_id, source, or text."
            )

    return index, metadata


def embed_question(model: SentenceTransformer, question: str) -> np.ndarray:
    embedding = np.asarray(
        model.encode(question, convert_to_numpy=True),
        dtype=np.float32,
    )
    if embedding.shape != (EXPECTED_DIMENSIONS,):
        raise ValueError(
            f"Expected a {EXPECTED_DIMENSIONS}-dimensional question embedding, "
            f"got shape {embedding.shape}."
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Question embedding contains non-finite values.")

    query = np.ascontiguousarray(embedding.reshape(1, -1))
    if np.linalg.norm(query) == 0:
        raise ValueError("Question embedding has zero length.")
    faiss.normalize_L2(query)
    return query


def print_results(
    question: str,
    scores: np.ndarray,
    positions: np.ndarray,
    metadata: list[dict[str, object]],
) -> None:
    print(f"\nQuestion: {question}")
    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        if position < 0 or position >= len(metadata):
            raise ValueError(f"FAISS returned invalid vector position {position}.")

        record = metadata[int(position)]
        print(f"\nRank: {rank}")
        print(f"Chunk ID: {record['chunk_id']}")
        print(f"Source: {record['source']}")
        print(f"Cosine similarity: {float(score):.4f}")
        print(f"Text:\n{record['text']}")


def main() -> None:
    args = parse_args()
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer(MODEL_NAME)

    print("Dense vector retrieval ready. Type 'exit' to quit.")
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

        query = embed_question(model, question)
        scores, positions = index.search(query, args.top_k)
        print_results(question, scores, positions, metadata)


if __name__ == "__main__":
    main()
