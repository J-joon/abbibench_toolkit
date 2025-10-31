from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import json
import functools
from functools import cache
from scipy.stats import spearmanr
from typing import TypeAlias, Literal

dataset_name_t: TypeAlias = Literal[
        '1mlc', '1mlc_LC', '1n8z', '2fjg', '3gbn', '4fqi', 'aayl49', 'aayl49_ML', 'aayl51', '1mhp', '4d5_her2', '5a12_ang2', '5a12_vegf', 'aayl50_LC', 'aayl52_LC', 'g6_LC'
        ]

@dataclass(frozen=True, slots=False)
class DatasetConfig:
    metadata_path: Path
    dataset_name: dataset_name_t

    @property
    @functools.cache
    def metadata(self,) -> dict:
        with open(self.metadata_path, 'r') as file:
            metadata = json.load(file)
        return metadata

    @property
    @functools.cache
    def info(self,)->dict:
        return self.metadata[self.dataset_name]

    @property
    @functools.cache
    def dataset(self,)->pd.DataFrame:
        pdb_info = self.info
        df = pd.read_csv(pdb_info["affinity_data"][0])
        return df

    def compute_correlation(self, log_likelihood: pd.DataFrame)->tuple[float, float]: #return rho, p-value
        qdf = log_likelihood.dropna(subset=['log-likelihood'])
        return spearmanr(self.dataset['binding_score'], qdf['log-likelihood'])

    @property
    @cache
    def pdb(self,)->str:
        return self.dataset.info["pdb"]

    @property
    @cache
    def pdb_path(self,)->str:
        return self.dataset.info["pdb_path"]

    @property
    @cache
    def heavy_chain(self,)->str:
        return self.dataset.info["heavy_chain"]

    @property
    @cache
    def light_chain(self,)->str:
        return self.dataset.info["light_chain"]

    @property
    @cache
    def antigen_chains(self,)->tuple[str,]:
        return tuple(self.dataset.info["antigen_chains"])

    @property
    @cache
    def affinity_data(self,)->tuple[str,]:
        return tuple(self.dataset.info["affinity_data"])

    @property
    @cache
    def receptor_chains(self,)->tuple[str,]:
        return tuple(self.dataset.info["receptor_chains"])

    @property
    @cache
    def ligand_chains(self,)->tuple[str,]:
        return tuple(self.dataset.info["ligand_chains"])

    @property
    @cache
    def chain_order(self,)->tuple[str,]:
        return tuple(self.dataset.info["chain_order"])

    @property
    @cache
    def epitope_chain(self,)->str:
        return self.dataset.info["epitope_chain"]

    @property
    @cache
    def paratope_chain(self,)->str:
        return self.dataset.info["paratope_chain"]

