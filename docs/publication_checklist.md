# Public release checklist

## Current public-branch scope

This same-repository `github-public` branch is being prepared for review. It
contains source code, tests, documentation, and `.env.example`. It does not
contain a project license, source corpus, evaluation datasets, corpus-derived
artifacts, model checkpoints, or raw historical result reports. The sanitized
aggregate `reports/final_rag_evaluation_report.md` is allow-listed for
publication; it summarizes historical metrics without including raw case
data, credentials, or local machine paths.

## Items still requiring review

| Asset | Public-branch status | Remaining review |
|---|---|---|
| Original source corpus and derived chunks, embeddings, index, and metadata | Excluded | Confirm source ownership/license and rights to publish derived content before adding. |
| Retrieval, generation, pipeline, and safety datasets | Excluded | Confirm authorship, evidence/source permissions, and any third-party text. |
| Raw historical evaluation reports | Excluded | Review each report's input data, model/provider terms, and private/machine-specific content before adding. |
| Sanitized aggregate evaluation report | `reports/final_rag_evaluation_report.md` is allow-listed | Recheck that it contains only aggregate historical results and no private data before staging. |
| Ollama and embedding model weights | Downloaded outside this project | Review each model's license and usage restrictions separately. |
| Provider credentials | `.env` excluded; `.env.example` has placeholders | Rotate keys exposed in prior audit output; never commit or paste their values. |
| Project license | Not present | Select one only after confirming the right to license the included code and materials. |

## Before pushing

- Inspect the exact staged file list and tree for `github-public`; push only that
  branch, never all refs.
- Scan staged content and history for credentials, local paths, personal data,
  and large files.
- Confirm `.gitignore` still excludes raw reports while allowing only the
  reviewed aggregate report.
- Recheck `.gitignore` and upstream dataset/model terms.
- Add a project license only after confirming ownership and license scope.
- Obtain separate approval before configuring a remote or pushing.
