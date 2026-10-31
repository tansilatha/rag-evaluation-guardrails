import json
from pathlib import Path

import faiss
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
EMBEDDING_METADATA_PATH = DATA_DIR / "embedding_metadata.json"
INDEX_PATH = DATA_DIR / "faiss.index"
INDEX_METADATA_PATH = DATA_DIR / "faiss_metadata.json"
EXPECTED_SHAPE = (19, 384)
METADATA_FIELDS = ("chunk_id", "source", "text")


def load_chunk_records() -> list[dict[str, str]]:
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    if not records:
        raise ValueError(f"No chunks found in {CHUNKS_PATH}.")
    return records


def load_embedding_metadata() -> list[dict[str, str]]:
    with EMBEDDING_METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Embedding metadata must be a non-empty JSON array.")
    return metadata


def main() -> None:
    chunks = load_chunk_records()
    metadata = load_embedding_metadata()
    embeddings = np.load(EMBEDDINGS_PATH)

    if embeddings.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Expected embeddings with shape {EXPECTED_SHAPE}, got {embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values.")
    if len(chunks) != EXPECTED_SHAPE[0] or len(metadata) != EXPECTED_SHAPE[0]:
        raise ValueError(
            f"Expected {EXPECTED_SHAPE[0]} chunk and metadata records; "
            f"got {len(chunks)} chunks and {len(metadata)} metadata records."
        )

    expected_metadata = [
        {field: chunk.get(field) for field in METADATA_FIELDS}
        for chunk in chunks
    ]
    if metadata != expected_metadata:
        raise ValueError(
            "Embedding metadata does not match chunks.jsonl in content and order."
        )

    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero-length embedding.")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(INDEX_PATH))

    index_metadata = [
        {"vector_id": position, **chunk_metadata}
        for position, chunk_metadata in enumerate(metadata)
    ]
    with INDEX_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(index_metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    saved_index = faiss.read_index(str(INDEX_PATH))
    with INDEX_METADATA_PATH.open(encoding="utf-8") as file:
        saved_metadata = json.load(file)

    if saved_index.ntotal != len(chunks):
        raise ValueError(
            f"Saved index contains {saved_index.ntotal} vectors, "
            f"expected {len(chunks)}."
        )
    if saved_index.d != embeddings.shape[1]:
        raise ValueError(
            f"Saved index dimension is {saved_index.d}, "
            f"expected {embeddings.shape[1]}."
        )
    if [item.get("vector_id") for item in saved_metadata] != list(
        range(len(chunks))
    ):
        raise ValueError("Saved FAISS metadata positions are out of order.")
    if [
        {field: item.get(field) for field in METADATA_FIELDS}
        for item in saved_metadata
    ] != metadata:
        raise ValueError("Saved FAISS metadata does not match embedding order.")

    stored_vectors = saved_index.reconstruct_n(0, saved_index.ntotal)
    if not np.allclose(stored_vectors, vectors):
        raise ValueError("Saved FAISS vectors do not match the normalized inputs.")

    print(f"Index location: {INDEX_PATH}")
    print(f"Stored vectors: {saved_index.ntotal}")
    print(f"Vector dimensions: {saved_index.d}")
    print(f"Position metadata: {INDEX_METADATA_PATH}")


if __name__ == "__main__":
    main()
