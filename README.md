# AbBibench Toolkit — Model Integration Contract

## 1) Project Overview

**AbBibench Toolkit** is a lightweight evaluation and integration toolkit that reproduces (and makes easier to extend) a subset of the AbBiBench benchmarking flow.

The toolkit’s primary goals are:

* **Reproduce** the benchmark-style evaluation pipeline in a consistent, scriptable way.
* **Standardize** how new models are integrated (inputs/outputs, naming, and metadata).
* **Automate** leaderboard generation from model outputs, provided the model follows the contract defined below.

This repository provides *integration logic* and *tooling*; it does not claim authorship of the underlying benchmark.

## 2) Benchmark Backbone and Credits

This toolkit is **based on the logic and evaluation flow of AbBiBench**:

* Upstream benchmark repository: `MSBMI-SAFE/AbBiBench` ([https://github.com/MSBMI-SAFE/AbBiBench](https://github.com/MSBMI-SAFE/AbBiBench))

Important notes on credit and provenance:

* We are **not** the creators of AbBiBench, and we **must not** claim credit for inventing the benchmark.
* This toolkit does **not** necessarily vendor/copy the upstream repository verbatim.
* Instead, it **implements compatible evaluation logic and conventions** so that users can reproduce results and integrate new models more easily.

## 3) Design Choice: Why uv (and why conda may still exist)

### 3.1 Why uv is the preferred standard for new models

For *user-designed* and *newly integrated* models, this toolkit standardizes on **astral uv** ([https://docs.astral.sh/uv](https://docs.astral.sh/uv)) for Python dependency management because:

* **Fast, deterministic resolution** with lockfiles and reproducible installs.
* **Per-project isolation** is easy (e.g., each model can live in its own directory with its own `.venv`).
* **pyproject.toml-first** workflows align well with modern Python packaging.
* Encourages **clean dependency boundaries** between models.

### 3.2 Why conda may still be present in the ecosystem

Some upstream or legacy model implementations may depend on:

* non-Python system libraries,
* CUDA/PyTorch builds pinned to specific driver/toolkit combinations,
* older packages with fragile binary compatibility.

In those cases, **conda** can serve as a pragmatic “backbone compatibility layer.”

**Policy for this toolkit**:

* **Conda may be used** to support legacy/fragile environments.
* **New models should follow the uv-based contract** described below.
* The leaderboard/evaluation tooling is designed to consume outputs uniformly, regardless of how a model was executed—*as long as the output contract is satisfied*.

## 4) Contracts

This section defines the **minimum contract** a new model must satisfy so that:

1. inference can be run in a predictable way, and
2. results can be automatically incorporated into the leaderboard.

### 4.1 Definitions

* **Dataset ID**: The benchmark dataset identifier (e.g., `1mlc`, `2fjg`, `3gbn_h1`, `aayl49_ML`, etc.).
* **GT CSV**: Ground-truth affinity CSV under `data/binding_affinity/<dataset>_benchmarking_data.csv`.
* **Pred CSV**: A model output file produced under `./outputs/`.
* **Score column**: The column used for Spearman correlation vs `binding_score`.

### 4.2 File naming contract

Each model must write one CSV per dataset under:

* `./outputs/<dataset>_benchmarking_data_<MODEL>_scores.csv`

Where:

* `<dataset>` matches the dataset name in the ground-truth file.
* `<MODEL>` is the model identifier used in the leaderboard.

Examples:

* `outputs/2fjg_benchmarking_data_ProteinMPNN_scores.csv`
* `outputs/3gbn_h1_benchmarking_data_diffab_scores.csv`

### 4.3 Required columns contract

The output CSV **must** include:

* `binding_score` (copied through from the benchmark input rows)
* a **score column** used for correlation

By default, the toolkit expects the score column name:

* `log-likelihood`

However, the toolkit may support model-specific score columns (e.g., `dg`, `EpitopeSASA (mut)`) via explicit configuration.

### 4.4 Row alignment / join key contract

To compute correlation correctly, predictions must align with GT.

Supported alignment strategies:

**(A) Key-based merge (preferred)**

If available, include stable key columns that exist in the GT CSV, such as:

* `heavy_chain_seq`
* `light_chain_seq`

Then the toolkit will merge on the available keys.

**(B) Mutational datasets (sequence-only variants)**

Some datasets may use mutation-only sequences, e.g.:

* `mut_heavy_chain_seq`

If the GT and pred files use mutation columns, the contract requires that:

* the pred CSV includes the same mutation key column present in GT, OR
* the toolkit’s dataset-specific key mapping is configured to merge correctly.

**(C) Row-order fallback (discouraged)**

If no keys exist, the toolkit may fall back to row order—but this is fragile.

**Contract requirement**:

* If your pred CSV omits merge keys, you must guarantee identical row ordering to the GT CSV.

### 4.5 Correlation definition contract

Leaderboard uses:

* **Spearman rank correlation** between `binding_score` and the model score column.

Some models may require post-processing before correlation:

* **Negation** for metrics where “lower is better” (e.g., `FoldX`, `epitopeSA`).

If your model requires negation or uses a non-standard score column, you must declare it in the model configuration.

### 4.6 Minimal model package layout (uv)

A new model integration should be structured as:

```
models/<your_model>/
  pyproject.toml
  README.md
  src/<your_model>/...
  run.py
```

Where:

* `pyproject.toml` defines dependencies.
* `run.py` (or an equivalent CLI entrypoint) can execute inference and produce outputs in `./outputs/`.
* The command must be runnable via:

```
uv run python run.py --dataset <dataset_id>
```

### 4.7 “It just works” checklist

To be automatically picked up by leaderboard generation:

* Output file path matches the naming contract.
* CSV contains `binding_score` and `log-likelihood` (or configured score column).
* Rows align with GT via merge keys (preferred) or row order (fallback).
* No NaNs in the score column for evaluated rows.

---
## 5) Usage

This section describes the **standard user-facing commands** provided by the AbBibench Toolkit.
All commands are executed via **astral uv**, and assume a project layout that follows the contracts above.

install
```bash
uv pip install abbibench_toolkit@https://github.com/J-joon/abbibench_toolkit
```

### 5.1 Dataset Download


#### Command

```bash
uv run abbibench-toolkit-download-dataset
```

#### Behavior

* Downloads AbBiBench-compatible benchmark datasets.
* Stores files under:

```
<root_dir>/
  data/
    binding_affinity/
      <dataset>_benchmarking_data.csv
```

* Dataset IDs match those used by model execution and output naming.

---

### 5.2 Leaderboard Generation

#### Assumptions

```
<root_dir>/
  data/
    binding_affinity/
      <dataset>_benchmarking_data.csv
  outputs/
    <dataset>_benchmarking_data_<MODEL>_scores.csv
```

#### Command

```bash
uv run abbibench-toolkit-make-leaderboard
```

#### Behavior

* Scans `outputs/` for valid prediction CSVs.
* Matches predictions to GT data.
* Computes Spearman correlation.
* Applies model-specific post-processing if configured.
* Produces a consolidated leaderboard.

---


# EigenDrug Project - SNU Creative Integrated Design 1 (Fall 2025)

This project is part of the **Creative Integrated Design 1** course (Course Code: M1522.000200 001) in the **Department of Computer Science and Engineering, College of Engineering, Seoul National University** during the **Fall semester of 2025**.

The project is conducted in collaboration with **EigenDrug Inc.**, as a real-world industry assignment.

## Author

* **Name:** Jaejoon Kim
* **Email:** [jjkim030309@gmail.com](mailto:jjkim030309@gmail.com)
* Please contact via email for any inquiries or issues related to the project.

## Dependencies

* astral uv ([https://docs.astral.sh/uv](https://docs.astral.sh/uv))

