# Final RAG Evaluation Report

## 1. Project Overview

This report summarizes existing component evaluations and the completed
end-to-end RAG evaluation for this project. The values below come from saved
historical reports; no new model evaluations were run to prepare this summary.

The final pipeline run used the Aion Labs judge:

- Provider: `aion`
- Model: `aion-labs/aion-3.0-mini`
- Generator: Ollama `qwen2.5:3b`

API key values are intentionally omitted.

## 2. Architecture and Flow

1. The question is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
2. `retrieve.py` searches the existing normalized FAISS index.
3. `rag_chat.retrieve_chunks()` maps FAISS results to chunk IDs and text.
4. `rag_chat.generate_answer()` assembles the context and sends it to Ollama.
5. The generated answer is evaluated by the configured Aion judge through the shared `evaluate_generation.py` prompt, JSON parsing, validator, and bounded retry logic.

## 3. Configuration

| Component | Configuration |
|---|---|
| Chunking | `RecursiveCharacterTextSplitter`; 500-character chunks; 75-character configured overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`; 384 dimensions |
| Retriever | Existing normalized FAISS inner-product index; 19 vectors |
| Pipeline retrieval depth | `TOP_K=3` |
| Generator | Ollama `qwen2.5:3b` |
| Final judge provider | Aion Labs |
| Final judge model | `aion-labs/aion-3.0-mini` |
| Judge retry | One bounded retry for malformed/invalid responses only |

## 4. Component Results

| Component | Dataset / denominator | Actual result | Status |
|---|---:|---|---|
| Chunking | 19 chunks from one source document | 0 empty chunks; 234-484 characters; no non-whitespace omissions; observed overlap 0-69 characters | Completed |
| Embeddings | 10 evaluation questions; 19 document chunks | 384 dimensions; finite vectors; pairwise relevant-vs-unrelated accuracy `0.9964`; mean relevant similarity `0.6474` vs unrelated `0.2634`; average query embedding latency `15.20 ms` | Completed |
| Retrieval | 5 answerable retrieval goldens; answer-absent record excluded | @1: Hit Rate `0.800`, Recall `0.600`, Precision `0.800`, MRR `0.800`; @3: `1.000`, `1.000`, `0.4667`, `0.8667`; @5: `1.000`, `1.000`, `0.2800`, `0.8667`; average retrieval latency `30.30 ms` | Completed |
| Context assembly | 5 answerable retrieval goldens | All retrieved chunks reached the generator; 0 empty chunks; 0 duplicate cases; 0 omissions; maximum observed context `1,370` characters | Completed |
| Answer generation | 6 generation goldens | 6 records scored in the existing report; correctness `0.825`, faithfulness `0.975`, relevance `0.950`; average generation latency `17,526.74 ms` | Completed, prior OpenRouter report |
| End-to-end pipeline | 6 pipeline goldens; 5 retrieval-applicable | 6 evaluated and judged; 0 skipped; 0 failed; average end-to-end latency `15,698.28 ms`; average judge latency `24,686.54 ms` | Completed with Aion |
| Llama Guard input classification | 60 labeled inputs (35 unsafe, 25 safe) | Accuracy `0.9667`; precision `1.0000`; recall `0.9429`; F1 `0.9706`; TP=33, FP=0, TN=25, FN=2; average latency `4,190.79 ms` | Historical input-only evaluation |

The generation report predates the Aion switch and records OpenRouter with `dots-studio/dots-3-note-preview:free`. Its scores are therefore reported as a prior component evaluation, not as Aion generation results.

The Llama Guard result is from the historical input-classification report. It
does not represent output classification or live end-to-end guarded RAG
performance.

## 5. Final End-to-End Pipeline Results

Source: the existing local `pipeline_evaluation.json` report (raw JSON is not
included in the public branch).

- Golden records: `6`
- Evaluated questions: `6`
- Successfully judged: `6`
- Skipped questions: `0`
- Failed questions: `0`
- Pipeline failures: `0`
- Generation failures: `0`
- Judge failures: `0`

### Retrieval Metrics

Retrieval metrics use the 5 answerable records only. The answer-absent record was still run through the pipeline and judged for abstention, but has no relevant chunk labels and is excluded from retrieval aggregates.

- Hit Rate@3: `1.000`
- Precision@3: `0.4667`
- Recall@3: `1.000`
- MRR@3: `0.8667`
- Retrieval denominator: `5` answerable records

### Answer Metrics

Answer metrics use all `6` successfully judged records, including the answer-absent abstention case:

- Correctness: `0.8667`
- Faithfulness: `0.9333`
- Relevance: `0.9500`
- Answer-score denominator: `6` valid judge responses

### Latency

- Average end-to-end latency through answer generation: `15,698.28 ms`
- Average judge latency: `24,686.54 ms`

End-to-end latency is measured from retrieval start through the final generated answer. Judge processing is measured separately and is not included in end-to-end latency.

### Per-Question Status

All six records completed with `evaluation_status: evaluated`. There were no failure reasons in the final report.

The answer-absent record (`golden_0003`) received a valid abstention judgment and was excluded only from retrieval aggregates.

## 6. Tests

Focused tests completed during the project:

- Retrieval evaluator: `9/9` passed
- Embedding evaluator: `5/5` passed
- Context evaluator: `3/3` passed
- Pipeline evaluator: `2/2` passed
- Generation/judge focused tests after Aion integration: `23` passed, with one known pre-existing failure

The remaining known failure is `test_judge_model_reads_environment_override`. It expects a process-level `JUDGE_MODEL` value to override `.env`, while the project intentionally loads `.env` with `override=True`.

The offline Llama Guard unit tests use mocked Ollama responses. They cover
safe/unsafe and unexpected/error responses, input blocking before retrieval,
withholding unsafe generated answers, and report overwrite protection. These
are code-path tests, not live model evaluations.

## 7. Guardrail status and missing results

- The application checks input before retrieval, then checks generated output
  before returning an answer. Unsafe or failed checks block or withhold content.
- `src/evaluation/evaluate_llama_guard.py` evaluates labeled input examples.
- `src/evaluation/evaluate_llama_guard_output.py` expects a separate,
  human-labeled output dataset. That dataset is not present, so no output
  classification metrics have been measured.
- `src/evaluation/evaluate_llama_guard_e2e.py` exercises the actual guarded RAG
  workflow on labeled input cases. No live end-to-end safety report has been
  produced; mocked tests do not substitute for one.
- No persisted interactive application execution logs were found during the
  documentation audit.

## 8. Limitations

- The component evaluations use small datasets: 5 answerable retrieval records, 6 generation records, and 6 pipeline records.
- The answer-generation report uses an earlier OpenRouter judge run, while the final pipeline report uses Aion. These scores should not be compared as a controlled provider comparison.
- Aion judge scores are model-based estimates and may vary between runs.
- Retrieval precision is lower than recall at depth 3 because the pipeline retrieves additional non-relevant chunks for some questions.
- The historical Llama Guard result covers only the custom labeled input set; it does not establish safety on other input distributions or generated answers.

## 9. Recommended Improvements

1. Expand the generation and pipeline golden datasets before drawing broad performance conclusions.
2. Update the known environment-override test so the focused suite is fully green.
3. If provider comparisons are needed, rerun generation and pipeline evaluations with the same dataset and controlled provider configuration.
4. Investigate the lower-scoring answer for `golden_0004` if answer quality needs improvement; the final pipeline report contains its generated answer and judge scores.
5. Create and review a separate output-labeled dataset before measuring output classification metrics.
6. Run the end-to-end Guard evaluator and retain its JSON/Markdown report when authorized local datasets and model services are available.

## 10. Final Project Status

The final end-to-end RAG pipeline evaluation is complete with Aion:

- Retrieval, context assembly, generation, and judging completed for all 6 pipeline records.
- No pipeline, generation, or judge failures occurred.
- The final report is based on valid Aion judge responses and existing scoring logic.
- The existing Guard input evaluation achieved `96.67%` accuracy on 60 labeled inputs; output-label and live end-to-end Guard results remain unmeasured.
