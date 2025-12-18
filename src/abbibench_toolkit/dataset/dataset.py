from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Iterable, Optional, TypeAlias, Literal

import biotite.structure.io as bsio
import pandas as pd
from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from scipy.stats import spearmanr


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

dataset_name_t: TypeAlias = Literal[
    "1mlc",
    "1mlc_LC",
    "1n8z",
    "2fjg",
    "3gbn",
    "4fqi",
    "aayl49",
    "aayl49_ML",
    "aayl51",
    "1mhp",
    "4d5_her2",
    "5a12_ang2",
    "5a12_vegf",
    "aayl50_LC",
    "aayl52_LC",
    "g6_LC",
]


@dataclass(frozen=True)
class ComplexMeta:
    key: str
    pdb: str
    pdb_path: Path
    heavy_chain: str
    light_chain: str
    antigen_chains: list[str]
    affinity_data: list[Path]
    receptor_chains: list[str]
    ligand_chains: list[str]
    chain_order: list[str]
    epitope_chain: str
    paratope_chain: str


# ---------------------------------------------------------------------
# Main dataset wrapper
# ---------------------------------------------------------------------

class Dataset:
    """
    Wrapper around a locally cloned AbBibench dataset repository.

    Expected layout:
      <root_dir>/data/
        binding_affinity/*.csv
        complex_structure/*.pdb
        metadata.json
        ...

    Provides:
    - HF Dataset loading (all affinity CSVs concatenated)
    - filtering by antigen/base id
    - PDB loading via Biotite

    Added:
    - `config(dataset_name)` -> DatasetConfig
    """

    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.data_dir = self.root_dir / "data"
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Expected data dir at {self.data_dir}")

        self.binding_dir = self.data_dir / "binding_affinity"
        self.struct_dir = self.data_dir / "complex_structure"
        self.metadata_path = self.data_dir / "metadata.json"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json at {self.metadata_path}")

        self._meta_raw: dict[str, dict[str, Any]] = json.loads(self.metadata_path.read_text())
        self._meta: dict[str, ComplexMeta] = self._normalize_metadata(self._meta_raw)

    # -------------------------
    # Config entrypoint (your requested style)
    # -------------------------
    def config(self, dataset_name: dataset_name_t) -> "DatasetConfig":
        return DatasetConfig(metadata_path=self.metadata_path, dataset_name=dataset_name)

    # -------------------------
    # Metadata access
    # -------------------------
    def keys(self) -> list[str]:
        return sorted(self._meta.keys())

    def get_meta(self, key: str) -> ComplexMeta:
        return self._meta[key]

    def base_id(self, antigen_id: str) -> str:
        return antigen_id.split("_", 1)[0]

    def pdb_path_for(self, key_or_base_id: str) -> Path:
        if key_or_base_id in self._meta:
            return self._meta[key_or_base_id].pdb_path
        if key_or_base_id not in self._meta:
            raise KeyError(f"No metadata for '{key_or_base_id}'")
        return self._meta[key_or_base_id].pdb_path

    def chains_for(self, key_or_base_id: str) -> dict[str, Any]:
        m = self.get_meta(key_or_base_id)
        return {
            "heavy_chain": m.heavy_chain,
            "light_chain": m.light_chain,
            "antigen_chains": m.antigen_chains,
            "receptor_chains": m.receptor_chains,
            "ligand_chains": m.ligand_chains,
            "chain_order": m.chain_order,
            "epitope_chain": m.epitope_chain,
            "paratope_chain": m.paratope_chain,
        }

    # -------------------------
    # Affinity dataset (HF)
    # -------------------------
    def affinity_files(self, glob: str = "*_benchmarking_data.csv") -> list[Path]:
        if not self.binding_dir.exists():
            return []
        return sorted(self.binding_dir.glob(glob))

    def load_affinity(
        self,
        *,
        add_source_column: bool = True,
        add_antigen_columns: bool = True,
        csv_glob: str = "*_benchmarking_data.csv",
    ) -> HFDataset:
        csv_paths = self.affinity_files(csv_glob)
        if not csv_paths:
            raise FileNotFoundError(f"No affinity CSV files found under {self.binding_dir}")

        parts: list[HFDataset] = []
        for p in csv_paths:
            ds_part = load_dataset(
                "csv",
                data_files={"data": str(p)},
                split="data",
            )

            if add_source_column:
                ds_part = ds_part.add_column("_source_csv", [p.name] * len(ds_part))

            if add_antigen_columns:
                antigen_id = self._antigen_id_from_csv_name(p.name)
                base_id = self.base_id(antigen_id)
                ds_part = ds_part.add_column("antigen_id", [antigen_id] * len(ds_part))
                ds_part = ds_part.add_column("base_id", [base_id] * len(ds_part))

            parts.append(ds_part)

        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    def filter_by_antigen(self, ds: HFDataset, antigen_id: str) -> HFDataset:
        if "antigen_id" not in ds.column_names:
            raise KeyError("Column 'antigen_id' missing. Load with add_antigen_columns=True.")
        return ds.filter(lambda x: x["antigen_id"] == antigen_id)

    def filter_by_base_id(self, ds: HFDataset, base_id: str) -> HFDataset:
        if "base_id" not in ds.column_names:
            raise KeyError("Column 'base_id' missing. Load with add_antigen_columns=True.")
        return ds.filter(lambda x: x["base_id"] == base_id)

    # -------------------------
    # Structure loading
    # -------------------------
    def load_structure(
        self,
        *,
        antigen_id: str,
        chains: Optional[Iterable[str]] = None,
    ):
        base = self.base_id(antigen_id)
        pdb_path = self.pdb_path_for(base)
        atom_array = bsio.load_structure(str(pdb_path))

        if chains is not None:
            chain_set = set(chains)
            mask = [c in chain_set for c in atom_array.chain_id]
            atom_array = atom_array[mask]

        return atom_array

    # -------------------------
    # Internal helpers
    # -------------------------
    def _antigen_id_from_csv_name(self, name: str) -> str:
        suffix = "_benchmarking_data.csv"
        if not name.endswith(suffix):
            raise ValueError(f"Unexpected affinity csv name: {name!r}")
        return name[: -len(suffix)]

    def _normalize_metadata(self, raw: dict[str, dict[str, Any]]) -> dict[str, ComplexMeta]:
        out: dict[str, ComplexMeta] = {}
        for key, v in raw.items():
            pdb_path = self._resolve_repo_relative_path(v["pdb_path"])
            affinity_paths = [self._resolve_repo_relative_path(p) for p in v.get("affinity_data", [])]

            out[key] = ComplexMeta(
                key=key,
                pdb=str(v["pdb"]),
                pdb_path=pdb_path,
                heavy_chain=str(v["heavy_chain"]),
                light_chain=str(v["light_chain"]),
                antigen_chains=list(v.get("antigen_chains", [])),
                affinity_data=affinity_paths,
                receptor_chains=list(v.get("receptor_chains", [])),
                ligand_chains=list(v.get("ligand_chains", [])),
                chain_order=list(v.get("chain_order", [])),
                epitope_chain=str(v.get("epitope_chain", "")),
                paratope_chain=str(v.get("paratope_chain", "")),
            )
        return out

    def _resolve_repo_relative_path(self, p: str) -> Path:
        pp = Path(p)
        parts = list(pp.parts)
        if len(parts) >= 2 and parts[0] == "." and parts[1] == "data":
            rel = Path(*parts[2:])
            return (self.data_dir / rel).resolve()
        if len(parts) >= 1 and parts[0] == "data":
            rel = Path(*parts[1:])
            return (self.data_dir / rel).resolve()
        return (self.root_dir / pp).resolve()


# ---------------------------------------------------------------------
# Per-dataset config (your requested correlation workflow)
# ---------------------------------------------------------------------

@dataclass(frozen=True, slots=False)
class DatasetConfig:
    """
    A lightweight view into one logical dataset name from metadata.json
    (e.g., '3gbn', '1mlc_LC', ...), plus helpers for evaluation.

    This assumes:
    - metadata.json paths point under ./data/...
    - affinity_data[0] is the CSV you want to evaluate against
    """

    metadata_path: Path
    dataset_name: dataset_name_t

    @property
    @cache
    def metadata(self) -> dict[str, Any]:
        with self.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @property
    @cache
    def info(self) -> dict[str, Any]:
        return self.metadata[self.dataset_name]

    @property
    @cache
    def affinity_csv_path(self) -> Path:
        # metadata.json stores strings like "./data/binding_affinity/....csv"
        p = Path(self.info["affinity_data"][0])
        parts = list(p.parts)
        # resolve relative to the repository root (parent of ./data/)
        repo_root = self.metadata_path.parent.parent  # <root>/data/metadata.json -> <root>
        if len(parts) >= 2 and parts[0] == "." and parts[1] == "data":
            return (repo_root / Path(*parts[1:])).resolve()
        if len(parts) >= 1 and parts[0] == "data":
            return (repo_root / p).resolve()
        return (repo_root / p).resolve()

    @property
    @cache
    def dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.affinity_csv_path)

    def compute_correlation(self, log_likelihood: pd.DataFrame) -> tuple[float, float]:
        """
        Compute Spearman correlation between ground-truth binding_score and predicted log-likelihood.

        Requirements:
        - self.dataset must have column: 'binding_score'
        - log_likelihood must have column: 'log-likelihood'

        IMPORTANT:
        - This computes correlation by ROW ORDER after dropping NaNs in log_likelihood.
          If your predictions are not in the exact same order as the CSV, you should merge
          using a stable key column instead (add that when available).
        """
        if "binding_score" not in self.dataset.columns:
            raise KeyError(f"'binding_score' not found in {self.affinity_csv_path}")

        if "log-likelihood" not in log_likelihood.columns:
            raise KeyError("'log-likelihood' column missing from log_likelihood DataFrame")

        qdf = log_likelihood.dropna(subset=["log-likelihood"])
        if len(qdf) != len(self.dataset):
            # still allow, but make mismatch explicit because it's a common silent error
            raise ValueError(
                f"Row count mismatch: gt={len(self.dataset)} vs pred(non-NaN)={len(qdf)}. "
                "If you intentionally have missing predictions, pass a prediction DF aligned to gt "
                "or implement a key-based merge."
            )

        rho, p = spearmanr(self.dataset["binding_score"], qdf["log-likelihood"])
        return float(rho), float(p)

    # ---- metadata accessors (cached) ----

    @property
    @cache
    def pdb(self) -> str:
        return str(self.info["pdb"])

    @property
    @cache
    def pdb_path(self) -> str:
        return str(self.info["pdb_path"])

    @property
    @cache
    def heavy_chain(self) -> str:
        return str(self.info["heavy_chain"])

    @property
    @cache
    def light_chain(self) -> str:
        return str(self.info["light_chain"])

    @property
    @cache
    def antigen_chains(self) -> tuple[str, ...]:
        return tuple(self.info["antigen_chains"])

    @property
    @cache
    def affinity_data(self) -> tuple[str, ...]:
        return tuple(self.info["affinity_data"])

    @property
    @cache
    def receptor_chains(self) -> tuple[str, ...]:
        return tuple(self.info["receptor_chains"])

    @property
    @cache
    def ligand_chains(self) -> tuple[str, ...]:
        return tuple(self.info["ligand_chains"])

    @property
    @cache
    def chain_order(self) -> tuple[str, ...]:
        return tuple(self.info["chain_order"])

    @property
    @cache
    def epitope_chain(self) -> str:
        return str(self.info["epitope_chain"])

    @property
    @cache
    def paratope_chain(self) -> str:
        return str(self.info["paratope_chain"])
