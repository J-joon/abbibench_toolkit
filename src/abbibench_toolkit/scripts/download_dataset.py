from huggingface_hub import list_repo_files, hf_hub_download
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import biotite.structure.io as bsio

REPO = "AbBibench/Antibody_Binding_Benchmark_Dataset"

# 1. List all CSV files in the binding_affinity directory
csv_files = [
    f for f in list_repo_files(REPO, repo_type="dataset")
    if f.startswith("binding_affinity/") and f.endswith("_benchmarking_data.csv")
]

# 2. Load and concatenate all subsets
all_splits = []
for csv in tqdm(csv_files, desc="Loading CSVs"):
    ds = load_dataset(REPO, data_files={ "data": csv }, split="train")
    all_splits.append(ds)
full_ds = concatenate_datasets(all_splits)
print(full_ds)    # overview of the full dataset

# 3. Filter for samples belonging to influenza H1 (3gbn_h1)
h1_ds = full_ds.filter(lambda x: x["antigen_id"].endswith("3gbn_h1"))

# 4. List PDB structure files corresponding to this antigen
antigen_id     = "3gbn_h1"
base_id        = antigen_id.split("_")[0]
structure_files = [
    f for f in list_repo_files(REPO, repo_type="dataset")
    if f.startswith(f"structures/{base_id}") and f.endswith(".pdb")
]

# 5. Download and parse each PDB using Biotite
for pdb_file in structure_files:
    local_pdb = hf_hub_download(
        repo_id=REPO, filename=pdb_file, repo_type="dataset"
    )
    print("Downloaded to:", local_pdb)
    atom_array = bsio.load_structure(local_pdb)
    print("Chains:", atom_array.chain_id)
