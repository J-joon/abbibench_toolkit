from typing import Protocol, runtime_checkable
from dataclasses import datacass
import pandas as pd
from .config.config import ModelConfig
from functools import cache

SUPPORTING_MODELS = (
        "diffab",
        "ProteinMPNN", #https://github.com/dauparas/ProteinMPNN

@runtime_checkable
class Model(Protocol):
    def get_log_likelihood(dataset: pd.DataFrame)->pd.DataFrame:...
