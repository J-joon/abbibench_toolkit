import tyro
from dataclasses import dataclass
from typing import Literal, TypeAlias, Optional, Protocol
from pathlib import Path
from ..model import diffab, protein_mpnn
from ..dataset.dataset import DatasetConfig
import pandas as pd
from functools import cache


model_name_t: TypeAlias = Literal[
        "AntiBERTy",
        "Antifold",
        "CurrAb",
        "ESM-2",
        "ESM-IF",
        "ESM3-Open",
        "MEAN",
        "ProGen2",
        "ProSST",
        "ProtGPT2",
        "ProteinMPNN",
        "SaProt",
        "diffab",
        "dyMEAN",
        ]


model_config_t: TypeAlias = diffab.Config

@dataclass(frozen=True, slots=False)
class Config:
    data_config: DatasetConfig
    model: model_name_t
    output_dir: Path
    model_config: model_config_t
    is_correlation_only: bool = False

    @property
    @cache
    def output_path(self,)->Path:
        return self.output_dir / f"{self.data_config.dataset_name}_benchmarking_data_{self.model}_scores.csv"

    def save_result(self, result: pd.DataFrame):
        result.to_csv(self.output_path, header=True, index=False)

    @property
    @cache
    def result(self,)->pd.DataFrame:
        return pd.read_csv(self.output_path)
