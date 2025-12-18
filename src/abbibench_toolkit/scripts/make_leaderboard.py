from __future__ import annotations

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

# Dataset order (matches the official table order)
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

# Special target columns for specific models
SPECIAL_COLUMNS: Dict[str, str] = {
    "FoldX": "dg",
    "epitopeSA": "EpitopeSASA (mut)",
}

# Fixbb models and their source columns
FIXBB_SOURCES: Dict[str, Tuple[str, str]] = {
    "MEAN_fixbb": ("MEAN", "log-likelihood (fixed backbone)"),
    "dyMEAN_fixbb": ("dyMEAN", "log-likelihood (fixed backbone)"),
    "diffab_fixbb": ("diffab", "log-likelihood (fixed backbone)"),
}

# Models whose target values are negated before correlation
NEGATE_MODELS = {"epitopeSA", "FoldX"}


# -----------------------------
# Normalization helpers
# -----------------------------

_PREFIX_RE = re.compile(r"^(AHL|HLA|LAH)_")

def normalize_model_name(model_part: str) -> str:
    """Drop chain-order prefixes (AHL_/HLA_/LAH_) to match official keys."""
    return _PREFIX_RE.sub("", model_part)


def normalize_dataset_name(dataset_part: str) -> str:
    """Map local dataset ids to official dataset ids."""
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


# -----------------------------
# Parsing + metrics
# -----------------------------

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
    """Match official behavior: pandas corr(method='spearman') after dropping NaNs."""
    tmp = pd.concat([binding_score, target_values], axis=1).dropna()
    if len(tmp) == 0:
        return float("nan")
    return float(tmp.corr(method="spearman").iloc[0, 1])


def _format_float(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (float, np.floating, int, np.integer)):
        return f"{float(x):.2f}"
    return str(x)


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """
    Lightweight markdown table renderer (no external dependency like tabulate).
    Assumes df contains display-ready values (strings or numbers).
    """
    cols = list(df.columns)
    rows = [[_format_float(v) for v in df.iloc[i].tolist()] for i in range(len(df))]

    # widths
    col_widths = []
    for j, c in enumerate(cols):
        w = len(str(c))
        for r in rows:
            w = max(w, len(str(r[j])))
        col_widths.append(w)

    def _row_line(xs: List[str]) -> str:
        parts = [f" {str(xs[j]).ljust(col_widths[j])} " for j in range(len(xs))]
        return "|" + "|".join(parts) + "|"

    header = _row_line([str(c) for c in cols])
    sep = "|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|"
    body = "\n".join(_row_line(r) for r in rows)
    return "\n".join([header, sep, body]) + "\n"


# -----------------------------
# CLI
# -----------------------------

@dataclass(frozen=True)
class Args:
    # Root directory containing {data, outputs}. Artifacts will be written to <root>/artifacts by default.
    root_dir: Path = Path(".")

    # Optional override for outputs directory; if not set, uses <root>/outputs
    outputs_dir: Optional[Path] = None

    # If provided, override dataset order
    dataset_order: Sequence[str] = tuple(DEFAULT_DATASET_ORDER)

    # Drop any model row that has NaN in ANY dataset column
    drop_rows_with_nan: bool = True

    # If True, include non-official models found in outputs as well (default: official-only)
    include_nonofficial: bool = False

    # If True, include datasets not in dataset_order (appended after the ordered ones)
    include_unknown_datasets: bool = False

    # Optional output overrides; if not set, uses <root>/artifacts/leaderboard.{csv,md}
    out_csv: Optional[Path] = None
    out_md: Optional[Path] = None

    # Print extra diagnostics
    verbose: bool = False


def _resolve_paths(a: Args) -> Tuple[Path, Path, Path, Path]:
    root = a.root_dir.resolve()
    outputs = a.outputs_dir.resolve() if a.outputs_dir else (root / "outputs")
    artifacts = root / "artifacts"
    out_csv = a.out_csv.resolve() if a.out_csv else (artifacts / "leaderboard.csv")
    out_md = a.out_md.resolve() if a.out_md else (artifacts / "leaderboard.md")
    return root, outputs, out_csv, out_md


def main(a: Args) -> None:
    root, outputs_dir, out_csv, out_md = _resolve_paths(a)
    dataset_order = list(a.dataset_order)

    if not outputs_dir.exists():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    # Official model set
    official_models: List[str] = []
    for _group, ms in OFFICIAL_GROUPS.items():
        official_models.extend(ms)
    official_models_set = set(official_models)

    # Cache loaded CSVs keyed by (normalized_dataset, normalized_model) -> (path, df)
    # Use newest mtime if duplicates exist.
    csv_cache: Dict[Tuple[str, str], Tuple[Path, pd.DataFrame]] = {}

    csv_paths = sorted(outputs_dir.rglob("*.csv"))
    if a.verbose:
        print(f"[INFO] root_dir     : {root}")
        print(f"[INFO] outputs_dir  : {outputs_dir}")
        print(f"[INFO] found csv    : {len(csv_paths)} files")

    unknown_datasets_found: List[str] = []

    for path in csv_paths:
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue

        dataset_part, model_part = parsed
        dataset_name = normalize_dataset_name(dataset_part)
        model_name = normalize_model_name(model_part)

        # Dataset filtering
        if dataset_name not in dataset_order:
            if a.include_unknown_datasets:
                if dataset_name not in unknown_datasets_found:
                    unknown_datasets_found.append(dataset_name)
            else:
                continue

        # Model filtering (official-only by default)
        if (not a.include_nonofficial) and (model_name not in official_models_set):
            # allow base models needed for fixbb only if they are official anyway; this is mainly for safety.
            continue

        try:
            df = pd.read_csv(path)
        except Exception as e:
            if a.verbose:
                print(f"[WARN] failed to read CSV: {path} ({e})")
            continue

        key = (dataset_name, model_name)
        if key in csv_cache:
            old_path, _old_df = csv_cache[key]
            if path.stat().st_mtime <= old_path.stat().st_mtime:
                continue  # keep newer
        csv_cache[key] = (path, df)

    # Extend dataset order if unknown datasets are allowed
    if a.include_unknown_datasets and unknown_datasets_found:
        # preserve stable order for unknowns: sort lexicographically and append
        for d in sorted(unknown_datasets_found):
            if d not in dataset_order:
                dataset_order.append(d)

    # Compute correlations
    # data[dataset][model] = corr
    data: Dict[str, Dict[str, float]] = {}

    for (dataset_name, model_name), (_path, df) in csv_cache.items():
        data.setdefault(dataset_name, {})

        # Determine target column
        target_col = "log-likelihood"
        if model_name in SPECIAL_COLUMNS:
            target_col = SPECIAL_COLUMNS[model_name]

        # Robust column match
        if model_name == "FoldX":
            target_col = _first_existing_column(df, ["dg", "DG", "delta_g", "deltaG", "dG"]) or target_col
        elif model_name == "epitopeSA":
            target_col = _first_existing_column(
                df,
                ["EpitopeSASA (mut)", "EpitopeSASA(mut)", "epitope_sasa_mut"],
            ) or target_col
        else:
            target_col = _first_existing_column(df, ["log-likelihood", "log_likelihood"]) or target_col

        # Fixbb handled later
        if model_name in FIXBB_SOURCES:
            continue

        if "binding_score" in df.columns and target_col in df.columns:
            binding_score = df["binding_score"]
            target_values = df[target_col]

            if model_name in NEGATE_MODELS:
                target_values = -target_values

            corr = _spearman_corr(binding_score, target_values)
            data[dataset_name][model_name] = corr
        else:
            data[dataset_name][model_name] = float("nan")

    # Handle fixbb models separately
    for fixbb_model, (base_model, ll_col) in FIXBB_SOURCES.items():
        for dataset_name in dataset_order:
            key = (dataset_name, base_model)
            if key not in csv_cache:
                continue

            _path, df = csv_cache[key]
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
                data.setdefault(dataset_name, {})
                data[dataset_name][fixbb_model] = corr

    # Convert to DataFrame (datasets as rows)
    result_df = pd.DataFrame.from_dict(data, orient="index")

    # Ensure all official model columns exist and maintain official order
    ordered_columns: List[str] = []
    for _group, cols in OFFICIAL_GROUPS.items():
        ordered_columns.extend(cols)

    # If include_nonofficial, keep any extra columns too (appended after official)
    extra_cols: List[str] = []
    if a.include_nonofficial:
        extra_cols = [c for c in result_df.columns if c not in ordered_columns]
        extra_cols = sorted(extra_cols)

    for col in ordered_columns + extra_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan

    # Reorder columns and datasets
    result_df = result_df[ordered_columns + extra_cols]
    result_df = result_df.reindex(dataset_order)

    # Build leaderboard table (models as rows, datasets as columns)
    table = result_df.transpose()  # index=model, columns=dataset

    # Keep only rows we want
    if a.include_nonofficial:
        keep_rows = [m for m in (ordered_columns + extra_cols) if m in table.index]
    else:
        keep_rows = [m for m in ordered_columns if m in table.index]
    table = table.loc[keep_rows]

    # Drop rows with any NaN (if requested)
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index(drop=True).to_csv(out_csv, index=False)

    # Write Markdown (pretty formatting)
    md = table.reset_index(drop=True).copy()
    for c in dataset_order:
        if c in md.columns:
            md[c] = md[c].map(_format_float)
    md["Avg. Spearman ↑"] = md["Avg. Spearman ↑"].map(_format_float)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    try:
        # may require 'tabulate' depending on pandas build
        md_text = md.to_markdown(index=False)
    except Exception:
        md_text = _df_to_markdown_simple(md)
    out_md.write_text(md_text, encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")

    if a.verbose:
        print()
        print(md.to_string(index=False))


def entrypoint() -> None:
    main(tyro.cli(Args))


if __name__ == "__main__":
    entrypoint()

