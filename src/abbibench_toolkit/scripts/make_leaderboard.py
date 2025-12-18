from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tyro


# -----------------------------
# Official leaderboard config
# -----------------------------

# Official groups (normalized model names; we strip AHL_/HLA_/LAH_ prefixes)
OFFICIAL_GROUPS: Dict[str, List[str]] = {
    "Masked LM": ["ESM2", "AntiBERTy", "CurrAb", "SaProt", "ProSST", "ESM3-Open-structure"],
    "Autoregressive LM": ["progen2-large", "ProtGPT2"],
    "Inverse Folding": ["ProteinMPNN", "ESMIF1", "Antifold"],
    "Diffusion": ["diffab", "diffab_fixbb"],
    "Graph Model": ["MEAN", "MEAN_fixbb", "dyMEAN", "dyMEAN_fixbb"],
    "Biophysics": ["epitopeSA", "FoldX"],
}

# Dataset order (this matches the wider official table you pasted earlier)
DEFAULT_DATASET_ORDER: List[str] = [
    "1mhp",
    "1mlc",
    "1n8z",
    "2fjg",
    "3gbn_h1",
    "3gbn_h9",
    "4fqi_h1",
    "4fqi_h3",
    "5a12_ang2",
    "aayl50",
    "aayl49",
    "aayl49_ML",
    "aayl51",
    "aayl52",
]

# Special target columns for specific models (official script)
SPECIAL_COLUMNS: Dict[str, str] = {
    "FoldX": "dg",
    "epitopeSA": "EpitopeSASA (mut)",
}

# Fixbb models and their source columns (official script)
FIXBB_SOURCES: Dict[str, Tuple[str, str]] = {
    "MEAN_fixbb": ("MEAN", "log-likelihood (fixed backbone)"),
    "dyMEAN_fixbb": ("dyMEAN", "log-likelihood (fixed backbone)"),
    "diffab_fixbb": ("diffab", "log-likelihood (fixed backbone)"),
}

# Models whose target values are negated before correlation (official script)
NEGATE_MODELS = {"epitopeSA", "FoldX"}


# -----------------------------
# Normalization helpers
# -----------------------------

_PREFIX_RE = re.compile(r"^(AHL|HLA|LAH)_")

def normalize_model_name(model_part: str) -> str:
    """
    Make local filenames compatible with official leaderboard keys:
    - Drop chain-order prefixes: AHL_/HLA_/LAH_
    - Keep the rest as-is
    """
    return _PREFIX_RE.sub("", model_part)


def normalize_dataset_name(dataset_part: str) -> str:
    """
    Map local dataset ids to official dataset ids.
    """
    aliases = {
        "1mhp_LC": "1mhp",
        "aayl50_LC": "aayl50",
        "aayl52_LC": "aayl52",
    }
    return aliases.get(dataset_part, dataset_part)


def infer_model_type(normalized_model: str) -> str:
    for group, models in OFFICIAL_GROUPS.items():
        if normalized_model in models:
            return group
    return "Other"


def medal_rank(i: int) -> str:
    if i == 1:
        return f"🥇 {i}"
    if i == 2:
        return f"🥈 {i}"
    if i == 3:
        return f"🥉 {i}"
    return str(i)


def _parse_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse:
      <dataset>_benchmarking_data_<model>_scores.csv
      <dataset>_benchmarking_data_<model>-scores.csv
    Return (dataset_part, model_part) or None if not match.
    """
    if "_benchmarking_data_" not in filename:
        return None

    dataset_part, model_part = filename.split("_benchmarking_data_", 1)

    if model_part.endswith("_scores.csv"):
        model_part = model_part[: -len("_scores.csv")]
    elif model_part.endswith("-scores.csv"):
        model_part = model_part[: -len("-scores.csv")]
    else:
        return None

    return dataset_part, model_part


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _spearman_corr(binding_score: pd.Series, target_values: pd.Series) -> float:
    """
    Match official script behavior: pandas corr(method='spearman').
    """
    tmp = pd.concat([binding_score, target_values], axis=1)
    # Pairwise drop NaNs is handled by corr, but we keep it explicit.
    tmp = tmp.dropna()
    if len(tmp) == 0:
        return float("nan")
    # If one column is constant, Spearman becomes NaN; keep as NaN.
    return float(tmp.corr(method="spearman").iloc[0, 1])


# -----------------------------
# CLI
# -----------------------------

@dataclass(frozen=True)
class Args:
    outputs_dir: Path = Path("./outputs")

    # If provided, override dataset order
    dataset_order: Sequence[str] = tuple(DEFAULT_DATASET_ORDER)

    # Drop any model row that has NaN in ANY dataset column
    drop_rows_with_nan: bool = True

    out_csv: Path = Path("./artifacts/leaderboard.csv")
    out_md: Path = Path("./artifacts/leaderboard.md")


def main(a: Args) -> None:
    outdir = a.outputs_dir.resolve()
    if not outdir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outdir}")

    dataset_order = list(a.dataset_order)

    # Official model set to keep (skip baselines)
    official_models: List[str] = []
    for _group, ms in OFFICIAL_GROUPS.items():
        official_models.extend(ms)
    official_models_set = set(official_models)

    # Cache loaded CSVs keyed by (normalized_dataset, normalized_model)
    # Note: multiple raw models may normalize to the same model (e.g., HLA_ProtGPT2 -> ProtGPT2).
    # For the official leaderboard reproduction, we take the exact matching normalized model entries.
    csv_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    for filename in os.listdir(outdir):
        if not filename.endswith(".csv"):
            continue
        parsed = _parse_filename(filename)
        if parsed is None:
            continue

        dataset_part, model_part = parsed
        dataset_name = normalize_dataset_name(dataset_part)
        model_name = normalize_model_name(model_part)

        # Keep only datasets we care about (optional but reduces noise)
        if dataset_name not in dataset_order:
            continue

        # Keep only official models (skip baselines)
        if model_name not in official_models_set and model_name not in FIXBB_SOURCES:
            # FIXBB_SOURCES are computed later, not directly from model_name files,
            # but allow base models to be loaded anyway.
            pass

        df = pd.read_csv(outdir / filename)
        # Use last-write-wins if duplicates exist; this is fine for your usage.
        csv_cache[(dataset_name, model_name)] = df

    # Compute correlations (official behavior)
    data: Dict[str, Dict[str, float]] = {}

    for (dataset_name, model_name), df in csv_cache.items():
        if dataset_name not in data:
            data[dataset_name] = {}

        # Determine target column exactly as official script (after normalization)
        target_col = "log-likelihood"
        if model_name in SPECIAL_COLUMNS:
            target_col = SPECIAL_COLUMNS[model_name]

        # Try to be slightly robust if your columns differ by minor naming
        if model_name == "FoldX":
            # official expects 'dg'
            target_col = _first_existing_column(df, ["dg", "DG", "delta_g", "deltaG", "dG"]) or target_col
        elif model_name == "epitopeSA":
            target_col = _first_existing_column(df, ["EpitopeSASA (mut)", "EpitopeSASA(mut)", "epitope_sasa_mut"]) or target_col
        else:
            target_col = _first_existing_column(df, ["log-likelihood", "log_likelihood"]) or target_col

        if model_name in FIXBB_SOURCES:
            # Fixbb handled later (official script)
            continue

        if "binding_score" in df.columns and target_col in df.columns:
            binding_score = df["binding_score"]
            target_values = df[target_col]

            if model_name in NEGATE_MODELS:
                target_values = -target_values

            corr = _spearman_corr(binding_score, target_values)
            data[dataset_name][model_name] = corr
        else:
            # Keep missing as NaN (no print spam by default)
            data[dataset_name][model_name] = float("nan")

    # Handle fixbb models separately (official script)
    for fixbb_model, (base_model, ll_col) in FIXBB_SOURCES.items():
        for dataset_name in dataset_order:
            if (dataset_name, base_model) not in csv_cache:
                continue
            df = csv_cache[(dataset_name, base_model)]

            # Robust column match for fixed-backbone LL
            ll_col_eff = _first_existing_column(
                df,
                [
                    ll_col,
                    "log-likelihood (fixed backbone)",
                    "log_likelihood (fixed backbone)",
                    "log-likelihood_fixed_backbone",
                    "log_likelihood_fixed_backbone",
                ],
            )
            if ll_col_eff is None:
                continue

            if "binding_score" in df.columns and ll_col_eff in df.columns:
                corr = _spearman_corr(df["binding_score"], df[ll_col_eff])
                if dataset_name not in data:
                    data[dataset_name] = {}
                data[dataset_name][fixbb_model] = corr

    # Convert to DataFrame like official script
    result_df = pd.DataFrame.from_dict(data, orient="index")

    # Ensure all official model columns exist
    ordered_columns: List[str] = []
    group_names: List[str] = []
    group_sizes: List[int] = []

    for group, cols in OFFICIAL_GROUPS.items():
        group_names.append(group)
        group_sizes.append(len(cols))
        ordered_columns.extend(cols)

    for col in ordered_columns:
        if col not in result_df.columns:
            result_df[col] = np.nan

    # Reorder and reindex datasets
    result_df = result_df[ordered_columns]
    result_df = result_df.reindex(dataset_order)

    # Build leaderboard table (models as rows, datasets as columns)
    table = result_df.transpose()  # index=model, columns=dataset

    # Keep only official models (and only those that appear at least once)
    table = table.loc[[m for m in ordered_columns if m in table.index]]

    # Drop rows with any NaN (as requested)
    if a.drop_rows_with_nan:
        table = table.dropna(axis=0, how="any")

    # Add Model Type and Avg
    table.insert(0, "Model", table.index)
    table.insert(0, "Model Type", [infer_model_type(m) for m in table.index])

    dataset_cols = [c for c in table.columns if c not in {"Model Type", "Model"}]
    table["Avg. Spearman ↑"] = table[dataset_cols].mean(axis=1, skipna=False)

    # Sort by Avg descending
    table = table.sort_values(by="Avg. Spearman ↑", ascending=False)

    # Rank
    table.insert(0, "Rank", [medal_rank(i) for i in range(1, len(table) + 1)])

    # Write CSV (raw floats)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index(drop=True).to_csv(a.out_csv, index=False)

    # Write Markdown (pretty formatting)
    md = table.reset_index(drop=True).copy()
    for c in dataset_order:
        if c in md.columns:
            md[c] = md[c].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    md["Avg. Spearman ↑"] = md["Avg. Spearman ↑"].map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")

    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.write_text(md.to_markdown(index=False), encoding="utf-8")

    print(f"Wrote: {a.out_csv}")
    print(f"Wrote: {a.out_md}")
    print()
    print(md.to_string(index=False))

def entrypoint():
    main(tyro.cli(Args))

if __name__ == "__main__":
    entryponint()
