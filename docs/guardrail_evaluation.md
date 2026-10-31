# Llama Guard 2 safety evaluation

## Implementation and guarded application flow

`src/guardrails/llama_guard.py` sends the user query and generated assistant
answer to the local Ollama model
`hf.co/QuantFactory/Meta-Llama-Guard-2-8B-GGUF:Q4_K_M`. The integration uses
distinct input and output instructions; the model was not fine-tuned in this
project.

`src/rag/rag_chat.py` enforces the sequence:

```text
User query
  -> Llama Guard input check
  -> retrieval (only if input is safe)
  -> answer generation
  -> Llama Guard output check
  -> final response (only if output is safe)
```

An unsafe input is blocked before retrieval. An unsafe generated answer is
withheld and is not included in the returned result. The application fails
closed for empty, unexpected, malformed, or failed Guard responses: they block
the request or withhold the answer. The offline unit tests mock Ollama and
verify these behaviors; they do not measure live-model performance.

## Historical input classification result

The existing input evaluation used 60 labeled examples (35 `UNSAFE`, 25
`SAFE`), all scored, with no request errors or unexpected responses. The saved
historical report records:

| Metric | Result |
|---|---:|
| Accuracy | **0.9667 (96.67%)** |
| Precision (UNSAFE positive class) | **1.0000** |
| Recall (UNSAFE positive class) | **0.9429** |
| F1-score | **0.9706** |
| False-positive rate | **0.0000** |
| False-negative rate | **0.0571** |
| Confusion counts | **TP 33, FP 0, TN 25, FN 2** |
| Average latency | **4,190.79 ms** |

Both false negatives were in `tricky_disguised_unsafe` (13 of 15 unsafe
examples in that category were classified unsafe). The report is an input
classification benchmark, not an output benchmark or a live evaluation of
guarded RAG end-to-end.

## Labeled datasets and evaluation scripts

The input evaluator expects `datasets/llama_guard_evaluation.json`. Each record
has `id`, `category`, `text`, and `expected_label` (`SAFE` or `UNSAFE`); an
optional `reason` records the label rationale. Keep labels human-reviewed.
The dataset is not included in the public branch.

Output evaluation expects a **separate** file at
`datasets/llama_guard_output_evaluation.json`; it is not present. Do not copy
input queries or invent output labels to fill it. Add only reviewed
assistant-response examples using the same record schema, with `text` holding
the assistant response and `expected_label` set by a human reviewer.

The evaluation modules are:

- `src/evaluation/evaluate_llama_guard.py`: labeled input classification.
- `src/evaluation/evaluate_llama_guard_output.py`: labeled output
  classification, using the separate output dataset.
- `src/evaluation/evaluate_llama_guard_e2e.py`: runs labeled input examples
  through the guarded RAG application and reports input classification and
  enforcement outcomes.

## Metrics and reports

The labeled input and output evaluators report accuracy, unsafe-positive
precision/recall/F1, confusion counts, false-positive and false-negative
counts/rates, per-case latency, and average latency. Input and output results
remain separate. By default, they write JSON and Markdown under
`reports/llama_guard/`:

- `input_safety_evaluation.json` and `.md`
- `output_safety_evaluation.json` and `.md`
- `end_to_end_safety_evaluation.json` and `.md`

The report writer refuses to overwrite an existing report file.

The end-to-end evaluator uses input dataset text as queries and invokes the
actual input-check, retrieval, generation, and output-check pipeline. It
reports input classification metrics, pipeline outcomes, and latency. It does
not claim output classification metrics because generated answers do not have
human ground-truth labels. Its `output_safe`/block counts describe observed
enforcement, not output accuracy or recall.

## What remains unmeasured

- There is no separate human-labeled output dataset, so no output precision,
  recall, F1, confusion matrix, or false-negative rate has been measured.
- The end-to-end evaluator exists, but no live guarded-RAG evaluation report
  has been produced. The current end-to-end safety tests use mocked pipeline
  results; they prove code-path behavior only.
- No persisted interactive application execution logs were found. Existing
  evidence is the historical input report and offline tests.
- The historical input dataset is small and custom; it is not a representative
  sample of all safe or unsafe requests. The observed metrics do not guarantee
  deployment safety.

## Commands

Run from the repository root after installing Python dependencies, starting
Ollama, pulling the Guard model, and supplying the required local inputs:

```powershell
# Labeled input classification; requires datasets/llama_guard_evaluation.json.
python -m src.evaluation.evaluate_llama_guard

# Labeled output classification; requires the separate labeled output dataset.
python -m src.evaluation.evaluate_llama_guard_output

# Guarded RAG flow; also requires the local index, metadata, and embedding model.
python -m src.evaluation.evaluate_llama_guard_e2e
```

All three commands call the local Ollama model. The end-to-end command also
uses the local RAG generator and may perform retrieval and generation for each
SAFE input. Evaluations were not run while preparing these documentation
updates. See the root README for setup and offline test instructions.
