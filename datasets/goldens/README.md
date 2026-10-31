# Golden datasets

This directory is reserved for retrieval, generation, and pipeline golden
datasets. Dataset files are not included in this public branch because their
source provenance and redistribution permissions have not been verified.

The generic candidate builder is in
[`src/evaluation/golden_dataset_builder.py`](../../src/evaluation/golden_dataset_builder.py).
Candidates require human review before they can be used to create evaluation
goldens. The builder refuses to overwrite existing golden files by default.
