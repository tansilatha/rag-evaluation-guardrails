import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts import generate_embeddings as embedding_generator
from src.rag import retrieve


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
QUESTIONS_PATH = PROJECT_DIR / "datasets" / "evaluation_questions.json"
REPORT_PATH = PROJECT_DIR / "reports" / "embedding_evaluation.json"
EXPECTED_DIMENSIONS = 384
PIPELINE_TOP_K = 5


def load_evaluation_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation questions not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Evaluation questions must be a non-empty JSON array.")
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Evaluation question {index} must be an object.")
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise ValueError(f"Evaluation question {index} is missing question text.")
        relevant = record.get("relevant_chunk_ids")
        if not isinstance(relevant, list) or not relevant or any(
            not isinstance(chunk_id, str) or not chunk_id.strip()
            for chunk_id in relevant
        ):
            raise ValueError(
                f"Evaluation question {index} must contain relevant_chunk_ids."
            )
    return records


def ensure_embedding_artifacts(regenerate: bool = False) -> bool:
    missing = not embedding_generator.EMBEDDINGS_PATH.is_file() or not embedding_generator.METADATA_PATH.is_file()
    if regenerate or missing:
        embedding_generator.main()
        return True
    return False


def load_saved_embeddings(chunks: list[dict[str, str]]) -> np.ndarray:
    if not embedding_generator.EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Embeddings not found: {embedding_generator.EMBEDDINGS_PATH}")
    if not embedding_generator.METADATA_PATH.is_file():
        raise FileNotFoundError(f"Embedding metadata not found: {embedding_generator.METADATA_PATH}")
    embeddings = np.asarray(np.load(embedding_generator.EMBEDDINGS_PATH), dtype=np.float32)
    metadata = json.loads(embedding_generator.METADATA_PATH.read_text(encoding="utf-8"))
    if embeddings.ndim != 2 or embeddings.shape != (len(chunks), EXPECTED_DIMENSIONS):
        raise ValueError(
            f"Expected embeddings with shape ({len(chunks)}, {EXPECTED_DIMENSIONS}), got {embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Saved embeddings contain non-finite values.")
    if not isinstance(metadata, list) or metadata != chunks:
        raise ValueError("Saved embedding metadata does not match current chunks.")
    if np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("Saved embeddings contain zero-length vectors.")
    return embeddings


def normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding batch contains a zero-length vector.")
    return embeddings / norms


def evaluate_semantic_separation(
    model: SentenceTransformer,
    questions: list[dict[str, Any]],
    chunks: list[dict[str, str]],
) -> dict[str, Any]:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]
    query_latencies = []
    query_results = []
    pairwise_correct = 0
    pairwise_total = 0
    positive_scores = []
    negative_scores = []

    for record in questions:
        started = time.perf_counter()
        query = np.asarray(
            model.encode(record["question"], convert_to_numpy=True), dtype=np.float32
        )
        query_latency = (time.perf_counter() - started) * 1000
        query_latencies.append(query_latency)
        if query.shape != (EXPECTED_DIMENSIONS,) or not np.isfinite(query).all():
            raise ValueError(f"Query embedding has invalid shape or values for: {record['question']}")
        query = query / np.linalg.norm(query)
        document_vectors = normalize_rows(
            np.asarray(model.encode(chunk_texts, show_progress_bar=False), dtype=np.float32)
        )
        scores = document_vectors @ query
        relevant_ids = set(record["relevant_chunk_ids"])
        relevant_scores = [
            float(scores[index]) for index, chunk_id in enumerate(chunk_ids) if chunk_id in relevant_ids
        ]
        unrelated_scores = [
            float(scores[index]) for index, chunk_id in enumerate(chunk_ids) if chunk_id not in relevant_ids
        ]
        for relevant_score in relevant_scores:
            for unrelated_score in unrelated_scores:
                pairwise_total += 1
                if relevant_score > unrelated_score:
                    pairwise_correct += 1
        positive_scores.extend(relevant_scores)
        negative_scores.extend(unrelated_scores)
        query_results.append(
            {
                "question": record["question"],
                "relevant_chunk_ids": record["relevant_chunk_ids"],
                "mean_relevant_similarity": statistics.mean(relevant_scores),
                "max_unrelated_similarity": max(unrelated_scores),
                "query_embedding_latency_ms": query_latency,
            }
        )

    return {
        "queries_evaluated": len(query_results),
        "query_embedding_dimensions_consistent": True,
        "query_embeddings_finite": True,
        "average_query_embedding_latency_ms": statistics.mean(query_latencies),
        "mean_relevant_similarity": statistics.mean(positive_scores),
        "mean_unrelated_similarity": statistics.mean(negative_scores),
        "relevant_vs_unrelated_pairwise_accuracy": (
            pairwise_correct / pairwise_total if pairwise_total else 0.0
        ),
        "query_results": query_results,
    }


def evaluate_retrieval_pipeline(
    model: SentenceTransformer,
    questions: list[dict[str, Any]],
) -> dict[str, Any]:
    index, metadata = retrieve.load_index_and_metadata()
    metadata_ids = [str(record["chunk_id"]) for record in metadata]
    hits_at_1 = []
    hits_at_k = []
    for record in questions:
        query = retrieve.embed_question(model, record["question"])
        _, positions = index.search(query, PIPELINE_TOP_K)
        retrieved = [metadata_ids[int(position)] for position in positions[0] if position >= 0]
        relevant = set(record["relevant_chunk_ids"])
        hits_at_1.append(bool(retrieved and retrieved[0] in relevant))
        hits_at_k.append(bool(set(retrieved) & relevant))
    return {
        "index_vectors": int(index.ntotal),
        "index_dimensions": int(index.d),
        "top_k": PIPELINE_TOP_K,
        "hit_rate_at_1": sum(hits_at_1) / len(hits_at_1),
        "hit_rate_at_k": sum(hits_at_k) / len(hits_at_k),
    }


def evaluate(
    report_path: Path = REPORT_PATH,
    regenerate: bool = False,
) -> dict[str, Any]:
    regenerated = ensure_embedding_artifacts(regenerate)
    chunks = embedding_generator.read_chunks()
    saved_embeddings = load_saved_embeddings(chunks)

    model_started = time.perf_counter()
    model = SentenceTransformer(embedding_generator.MODEL_NAME)
    model_loading_ms = (time.perf_counter() - model_started) * 1000

    document_started = time.perf_counter()
    fresh_documents = np.asarray(
        model.encode([chunk["text"] for chunk in chunks], show_progress_bar=False),
        dtype=np.float32,
    )
    document_generation_ms = (time.perf_counter() - document_started) * 1000
    if fresh_documents.shape != saved_embeddings.shape or not np.isfinite(fresh_documents).all():
        raise ValueError("Fresh document embeddings do not have the expected shape or values.")

    questions = load_evaluation_questions()
    semantic = evaluate_semantic_separation(model, questions, chunks)
    pipeline = evaluate_retrieval_pipeline(model, questions)
    report = {
        "model": embedding_generator.MODEL_NAME,
        "embedding_dimensions": EXPECTED_DIMENSIONS,
        "chunk_count": len(chunks),
        "evaluation_question_count": len(questions),
        "artifacts_regenerated": regenerated,
        "model_loading_time_ms": model_loading_ms,
        "document_embedding_generation_time_ms": document_generation_ms,
        "saved_embeddings_shape": list(saved_embeddings.shape),
        "saved_embeddings_finite": bool(np.isfinite(saved_embeddings).all()),
        "fresh_embeddings_shape": list(fresh_documents.shape),
        "fresh_embeddings_finite": bool(np.isfinite(fresh_documents).all()),
        "semantic_separation": semantic,
        "retrieval_pipeline": pipeline,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Model: {report['model']}")
    print(f"Chunks: {report['chunk_count']}")
    print(f"Dimensions: {report['embedding_dimensions']}")
    print(f"Model loading: {report['model_loading_time_ms']:.2f} ms")
    print(f"Document embedding generation: {report['document_embedding_generation_time_ms']:.2f} ms")
    print(f"Mean relevant similarity: {semantic['mean_relevant_similarity']:.4f}")
    print(f"Mean unrelated similarity: {semantic['mean_unrelated_similarity']:.4f}")
    print(f"Relevant-vs-unrelated pairwise accuracy: {semantic['relevant_vs_unrelated_pairwise_accuracy']:.4f}")
    print(f"Pipeline Hit Rate@1: {pipeline['hit_rate_at_1']:.4f}")
    print(f"Pipeline Hit Rate@{PIPELINE_TOP_K}: {pipeline['hit_rate_at_k']:.4f}")
    print(f"Report saved to: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the existing embedding model and retrieval compatibility.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate embedding artifacts using generate_embeddings.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args.report, args.regenerate)


if __name__ == "__main__":
    main()
