import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_DIR / "data" / "chunks.jsonl"
EMBEDDINGS_PATH = PROJECT_DIR / "data" / "embeddings.npy"
METADATA_PATH = PROJECT_DIR / "data" / "embedding_metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METADATA_FIELDS = ("chunk_id", "source", "text")


def read_chunks() -> list[dict[str, str]]:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_PATH}")

    chunks: list[dict[str, str]] = []
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL record on line {line_number}.")

            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Record on line {line_number} must be an object.")
            if any(not isinstance(record.get(field), str) for field in METADATA_FIELDS):
                raise ValueError(
                    f"Record on line {line_number} must contain string "
                    "chunk_id, source, and text fields."
                )
            if not record["chunk_id"] or not record["source"] or not record["text"]:
                raise ValueError(
                    f"Record on line {line_number} has an empty required field."
                )

            chunks.append({field: record[field] for field in METADATA_FIELDS})

    if not chunks:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs must be unique.")

    return chunks


def main() -> None:
    chunks = read_chunks()
    model = SentenceTransformer(MODEL_NAME)
    embeddings = np.asarray(
        model.encode([chunk["text"] for chunk in chunks], show_progress_bar=True),
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}.")
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Generated {embeddings.shape[0]} embeddings for {len(chunks)} chunks."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Generated embeddings contain non-finite values.")

    np.save(EMBEDDINGS_PATH, embeddings)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
        file.write("\n")

    saved_embeddings = np.load(EMBEDDINGS_PATH)
    with METADATA_PATH.open(encoding="utf-8") as file:
        saved_metadata = json.load(file)

    if saved_embeddings.shape[0] != len(saved_metadata):
        raise ValueError("Saved embedding and metadata counts do not match.")
    if [item["chunk_id"] for item in saved_metadata] != [
        chunk["chunk_id"] for chunk in chunks
    ]:
        raise ValueError("Saved metadata order does not match embedding row order.")
    if saved_metadata != chunks:
        raise ValueError("Saved metadata does not match the input chunk records.")
    if not np.isfinite(saved_embeddings).all():
        raise ValueError("Saved embeddings contain non-finite values.")

    print(f"Embeddings: {saved_embeddings.shape[0]}")
    print(f"Dimensions: {saved_embeddings.shape[1]}")
    print(f"All values finite: {bool(np.isfinite(saved_embeddings).all())}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
