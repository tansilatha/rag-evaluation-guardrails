# Implementation guide

## Application workflow

The application expects a locally supplied, authorized corpus. The public
branch does not contain the source document or generated retrieval artifacts.
The chunking utility in `scripts/chuck_document.py` uses LangChain's
`RecursiveCharacterTextSplitter` with a 500-character target and 75-character
configured overlap, then writes JSONL records under `data/`. The historical
filename `chuck_document.py` is intentionally retained.

`scripts/generate_embeddings.py` encodes chunk text with
`sentence-transformers/all-MiniLM-L6-v2` and saves vectors and metadata under
`data/`. The vectors have 384 dimensions. The index builder normalizes them and
stores them in a FAISS `IndexFlatIP` index; inner product between normalized
vectors is cosine similarity.

At chat time, `src/rag/retrieve.py` loads the locally prepared index and
metadata, encodes the question with the same embedding model, and returns
similar chunks. `src/rag/rag_chat.py` retrieves the top three chunks and sends
the question and retrieved text to Ollama's `qwen2.5:3b` chat endpoint. The
prompt asks the model to stay grounded in the context and abstain when evidence
is insufficient. The separate retrieval CLI defaults to five chunks.

## Guardrail behavior

`src/guardrails/llama_guard.py` sends the user query or generated assistant
response to the local Ollama model
`hf.co/QuantFactory/Meta-Llama-Guard-2-8B-GGUF:Q4_K_M`. Input and output use
different role-specific instructions.

`src/rag/rag_chat.py` implements the guarded sequence:

1. Classify the user query. An unsafe or failed input check blocks before
   retrieval.
2. Retrieve context and generate an answer for a safe input.
3. Classify the generated answer. An unsafe or failed output check withholds
   the answer instead of returning it.

The check is fail-closed for empty, unexpected, malformed, and failed model
responses. Llama Guard was not fine-tuned in this project.

## Evaluation workflows and historical results

Executable evaluation modules are in `src/evaluation/`. Evaluation datasets,
source documents, derived chunk/index artifacts, and raw historical reports
are local-only and are not included in the public branch.

The RAG evaluators cover embeddings, retrieval, context handoff, judged
generation quality, and a judged end-to-end RAG pipeline. The latest saved
historical results are summarized in
[`rag_evaluation.md`](./rag_evaluation.md) and
[`../reports/final_rag_evaluation_report.md`](../reports/final_rag_evaluation_report.md).
The historical Guard report covers input classification only: 60 labeled
cases, 96.67% accuracy, and 2 false negatives. These are existing results, not
new runs; the generation and pipeline scores used different judge providers.

The Guard evaluation scripts are:

- `src/evaluation/evaluate_llama_guard.py` — labeled input classification.
- `src/evaluation/evaluate_llama_guard_output.py` — labeled output
  classification; requires a separate human-labeled response dataset.
- `src/evaluation/evaluate_llama_guard_e2e.py` — runs labeled input examples
  through the guarded RAG flow and records input metrics and enforcement
  outcomes. It does not claim output classification metrics without labels for
  generated answers.

The output dataset and live end-to-end Guard results are not available. Offline
tests with mocked Ollama responses verify safety parsing, input blocking before
retrieval, answer withholding, and report overwrite protection; these tests
are not live model evaluations.

## Local commands

Run commands from the project root. Install Python dependencies from
`requirements.txt`; run Ollama separately and pull the model tags documented
in the root README. Supply an authorized local corpus first. Data-preparation
commands create or replace generated files under `data/`:

```powershell
python -m scripts.chuck_document
python -m scripts.generate_embeddings
python -m scripts.build_faiss_index
python -m src.rag.rag_chat
```

Core RAG evaluations:

```powershell
python -m src.evaluation.evaluate_embeddings
python -m src.evaluation.evaluate_retrieval
python -m src.evaluation.evaluate_context
python -m src.evaluation.evaluate_generation
python -m src.evaluation.evaluate_pipeline
```

Guard evaluations:

```powershell
# Historical input-classification workflow; requires the labeled input dataset.
python -m src.evaluation.evaluate_llama_guard

# Requires datasets/llama_guard_output_evaluation.json with human-reviewed labels.
python -m src.evaluation.evaluate_llama_guard_output

# Requires labeled input data, local index/metadata, embedding model, and Ollama.
python -m src.evaluation.evaluate_llama_guard_e2e
```

Generation, pipeline, and Guard commands make model calls. Generation and
pipeline runs may also call the configured external judge and incur costs.
Guard evaluators write separate reports under `reports/llama_guard/` and
refuse to overwrite existing report files. See the root README for setup and
the offline unit-test command.
