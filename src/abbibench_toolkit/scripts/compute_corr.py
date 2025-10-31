from abbibench_toolkit.dataset.dataset import DatasetConfig, dataset_name_t
import pandas as pd
import tyro
from pathlib import Path

def compute(model_name: str, dataset_name: dataset_name_t, metadata_path: Path = Path("data/metadata.json"), output_dir: Path = Path("outputs")):
    # 1. load dataset
    data_config = DatasetConfig(metadata_path, dataset_name)
    # 2. load result
    result_path = output_dir / f"{dataset_name}_benchmarking_data_{model_name}_scores.csv"
    log_likelihood = pd.read_csv(result_path)
    # 3. compute spearman correlation
    rho, p_value = data_config.compute_correlation(log_likelihood)
    print(f"rho: {rho:.4f}, p-value: {p_value:.4e}")
    return

def entrypoint():
    tyro.cli(compute)

if __name__=="__main__":
    entrypoint()
