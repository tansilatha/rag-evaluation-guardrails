# RAG evaluation notes

This page records the implemented evaluation methods and the existing
historical results. The metrics below are transcribed from local JSON reports;
the original datasets and raw reports are not included in the public branch.
No new evaluation was run to prepare this documentation.

## Historical results

| Evaluation | Dataset / denominator | Recorded result |
|---|---:|---|
| Embeddings | 10 evaluation questions; 19 document chunks | 384 dimensions; saved and fresh vectors finite; pairwise relevant-vs-unrelated accuracy **0.9964**; mean relevant similarity **0.6474**, unrelated **0.2634**; query embedding latency **15.20 ms** average |
| Retrieval component | 5 answerable retrieval goldens; one unanswerable record excluded | At k=1: Hit Rate **0.800**, Recall **0.600**, Precision **0.800**, MRR **0.800**. At k=3: Hit Rate **1.000**, Recall **1.000**, Precision **0.4667**, MRR **0.8667**. At k=5: Hit Rate **1.000**, Recall **1.000**, Precision **0.2800**, MRR **0.8667**. Average retrieval latency **30.30 ms** |
| Context handoff | 5 answerable goldens | All retrieved chunks and questions reached the generator; **0** empty contexts, duplicate cases, or omissions; maximum context **1,370 characters** |
| Generation component | 6 generation goldens | All 6 scored; correctness **0.825**, faithfulness **0.975**, relevance **0.950**; average generation latency **17,526.74 ms**; OpenRouter judge |
| End-to-end RAG pipeline | 6 pipeline goldens; retrieval metrics use 5 answerable records | 6 evaluated, 0 skipped, 0 failed. At k=3: Hit Rate **1.000**, Precision **0.4667**, Recall **1.000**, MRR **0.8667**. Answer scores: correctness **0.8667**, faithfulness **0.9333**, relevance **0.9500**. Average end-to-end latency **15,698.28 ms**; average judge latency **24,686.54 ms**; Aion judge |

The generation component report used OpenRouter model
`dots-studio/dots-3-note-preview:free`; the final pipeline report used Aion
model `aion-labs/aion-3.0-mini`. The scores are from distinct runs and are not a
controlled provider comparison. Judge scores are model-based assessments, not
human ground truth. The datasets are small and should not be treated as
representative estimates of general performance.

## Retrieval

`src/evaluation/evaluate_retrieval.py` evaluates a FAISS index and metadata
against labeled retrieval goldens using the MiniLM query encoder. It reports
Hit Rate, Precision, Recall, and MRR at configured K values. Unanswerable
records without relevant chunk IDs are skipped for retrieval aggregates.
Precision uses requested K as its denominator; retrieved chunk IDs are
deduplicated while preserving their first rank. The current implementation
does not calculate nDCG.

The standalone historical retrieval report used five answerable examples and
excluded one answer-absent record. The pipeline retrieval aggregates likewise
use five retrieval-applicable records from six total pipeline goldens.

## Embeddings and context assembly

`src/evaluation/evaluate_embeddings.py` validates saved and regenerated
embedding shapes and values, compares labeled and unrelated similarities,
measures pairwise ordering, and checks retrieval hit rates. The recorded
historical result used 19 chunks and 10 evaluation questions.

`src/evaluation/evaluate_context.py` intercepts the generator request to check
whether the question and retrieved evidence reach the prompt. This tests
context handoff, not whether the generated answer is faithful.

## Answer quality and end-to-end evaluation

`src/evaluation/evaluate_generation.py` scores correctness, faithfulness, and
relevance with a configured external LLM judge. It validates score ranges and
the expected structured response. `src/evaluation/evaluate_pipeline.py`
combines retrieval with answer generation and judging. Runs using different
providers or datasets are not controlled comparisons.

These RAG evaluation scripts call retrieval and answer generation directly;
they do not exercise the interactive `src/rag/rag_chat.py` input/output safety
checks. Guard classification and guarded RAG enforcement have separate
evaluators and are documented in
[`guardrail_evaluation.md`](./guardrail_evaluation.md).

## Golden dataset candidate builder

`src/evaluation/golden_dataset_builder.py` creates RAG golden candidates from
the authorized local source document and chunks. It validates that cited
evidence appears in the source and an actual chunk, then saves candidates with
`review_status: "pending"`. A human must review the question, answerability,
answer, category, and evidence before marking a candidate
`review_status: "approved"`. The builder creates retrieval, generation, and
pipeline goldens only when explicitly invoked with `--build-reviewed`; it
refuses to overwrite existing goldens unless `--overwrite-goldens` is passed.

The candidate builder is not an evaluation and does not certify semantic
correctness automatically. The source document, chunks, candidate logs, and
golden datasets are local inputs and are not included in this public branch.

## Reproduction

From the project root, after installing dependencies and supplying all required
authorized local datasets/artifacts and model services:

```powershell
python -m src.evaluation.evaluate_embeddings
python -m src.evaluation.evaluate_retrieval
python -m src.evaluation.evaluate_context
python -m src.evaluation.evaluate_generation
python -m src.evaluation.evaluate_pipeline
```

Generation and pipeline runs make model/judge calls and may vary or incur
provider costs. They write local reports; do not treat a new run as a
reproduction of an omitted historical report unless the data and configuration
are the same.
