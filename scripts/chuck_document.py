
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Define file paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DOCUMENT_PATH = PROJECT_DIR / "data" / "documents" / "machine_learning.txt"
OUTPUT_PATH = PROJECT_DIR / "data" / "chunks.jsonl"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75


def validate_chunks(source_text: str, chunks: list[Document]) -> None:
    if not chunks:
        raise ValueError("Chunking produced no chunks.")

    previous_start = -1
    previous_end = 0

    for index, document in enumerate(chunks):
        chunk_text = document.page_content
        chunk_start = document.metadata.get("start_index")
        chunk_end = (
            chunk_start + len(chunk_text)
            if isinstance(chunk_start, int)
            else -1
        )

        if not chunk_text:
            raise ValueError(f"Chunk {index} is empty.")
        if not isinstance(chunk_start, int):
            raise ValueError(f"Chunk {index} is missing a valid start index.")
        if chunk_start < 0 or chunk_end > len(source_text):
            raise ValueError(f"Chunk {index} has an invalid source range.")
        if chunk_start <= previous_start:
            raise ValueError(f"Chunk {index} is out of source order.")
        if len(chunk_text) > CHUNK_SIZE:
            raise ValueError(f"Chunk {index} exceeds {CHUNK_SIZE} characters.")
        if source_text[chunk_start : chunk_start + len(chunk_text)] != chunk_text:
            raise ValueError(
                f"Chunk {index} does not exactly match the source text; "
                "chunking may have altered its wording or spacing."
            )

        if chunk_start > previous_end:
            gap = source_text[previous_end:chunk_start]
            if gap.count("\n") < 2 or gap.strip("\r\n"):
                raise ValueError(
                    f"Chunk {index} skips more than a paragraph boundary at "
                    f"character {previous_end}."
                )
        elif index and previous_end - chunk_start > CHUNK_OVERLAP:
            raise ValueError(
                f"Chunk {index} overlaps the previous chunk by more than "
                f"{CHUNK_OVERLAP} characters."
            )

        previous_start = chunk_start
        previous_end = max(previous_end, chunk_end)

    if (
        previous_end < len(source_text)
        and not source_text[previous_end:].isspace()
    ):
        raise ValueError(
            f"Chunks omit non-whitespace source text starting at "
            f"character {previous_end}."
        )


# 2. Read the document
if not DOCUMENT_PATH.exists():
    raise FileNotFoundError(
        f"Document not found: {DOCUMENT_PATH}"
    )

text = DOCUMENT_PATH.read_text(encoding="utf-8")

if not text.strip():
    raise ValueError("The document is empty.")


# 3. Configure the chunking method
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end",
    length_function=len,
    strip_whitespace=True,
    add_start_index=True,
)


# 4. Split the document into chunks
chunks = splitter.create_documents([text])
validate_chunks(text, chunks)


# 5. Save chunks with metadata
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    for index, document in enumerate(chunks):
        record = {
            "chunk_id": f"chunk_{index:04d}",
            "source": DOCUMENT_PATH.name,
            "text": document.page_content,
        }

        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 6. Display the results
print("=" * 50)
print("CHUNKING RESULTS")
print("=" * 50)

print(f"Original document characters: {len(text)}")
print(f"Total chunks created: {len(chunks)}")
print(f"Chunk size: {CHUNK_SIZE} characters")
print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
print(f"Saved to: {OUTPUT_PATH}")

print("\nFIRST THREE CHUNKS")
print("=" * 50)

for index, document in enumerate(chunks[:3]):
    chunk_text = document.page_content
    print(f"\nChunk {index:04d}")
    print("-" * 30)
    print(chunk_text)
    print(f"\nCharacter count: {len(chunk_text)}")
