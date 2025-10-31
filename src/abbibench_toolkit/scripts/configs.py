from abbibench_toolkit.config.config import Config
from abbibench_toolkit.model import diffab
from abbibench_toolkit.dataset.dataset import DatasetConfig
from pathlib import Path

_CONFIGS = {
        "diffab_corr_only": (
            "diffab_corr_only", Config(
                DatasetConfig(
                    metadata_path = Path("Antibody_Binding_Benchmark_Dataset/metadata.json"),
                    dataset_name = "aayl49_ML",
                    ),
                "diffab",
                output_dir= Path("outputs"),
                model_config = diffab.Config(
                    opt_yml = "./models/diffab/configs/opt.yml",
                    fixbb_yml = "./models/diffab/configs/fixbb.yml",
                    opt_ckpt = "./models/diffab/trained_models/codesign_single.pt",
                    fixbb_ckpt = "./models/diffab/trained_models/fixbb.pt",
                    device = "cuda",
                    ),
                is_correlation_only = True,
                )
            )
        }
