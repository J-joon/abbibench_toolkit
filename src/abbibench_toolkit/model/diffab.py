from dataclasses import dataclass
import functools
from pathlib import Path
import torch
from functools import cache
import torch
from pandas import DataFrame


@dataclass(frozen=True, slots=False)
class Config:
    opt_yml: str
    fixbb_yml: str
    opt_ckpt: str
    fixbb_ckpt: str
    device: str

    @property
    def model(self,) -> tuple[torch.nn.Module, torch.nn.Module]:
        return (self.opt_model, self.fixbb_model)

    def load_model(checkpoint_path: str, device: str):
        ckpt = torch.load(config.checkpoint_path, map_location=config.device)
        model = get_model(ckpt['config'].model).to(device)
        model.load_state_dict(ckpt['model'])

    @property
    @functools.lru_cache(maxsize=1)
    def opt_model(self,):
        return self.load_model(self.opt_ckpt, self.device)

    @property
    @functools.lru_cache(maxsize=1)
    def fixbb_model(self,):
        return self.load_model(self.fxbb_ckpt, self.device)
