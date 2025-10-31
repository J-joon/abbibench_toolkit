import os
import torch
import random
import subprocess
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from ProteinMPNN.utils import load_structure, extract_coords_from_complex, get_metadata
from copy import deepcopy
from Bio.PDB import PDBParser, PDBIO
from Bio.Data import IUPACData
one_to_three = IUPACData.protein_letters_1to3_extended
three_to_one = IUPACData.protein_letters_3to1_extended
import shutil
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from abbibench_toolkit.dataset.dataset import DatasetConfig

@dataclass(frozen=True, slots=False)
class Config:
    dataset: DatasetConfig
    device: str
    mpnn_path: './models/ProteinMPNN'
    seed: int = 42

    @property
    def model(self,) -> tuple[torch.nn.Module, torch.nn.Module]:
        return (self.opt_model, self.fixbb_model)

    def load_model(checkpoint_path: str, device: str):
        ckpt = torch.load(config.checkpoint_path, map_location=config.device)
        model = get_model(ckpt['config'].model).to(device)
        model.load_state_dict(ckpt['model'])

    def inference(self,)->pd.DataFrame:
        setup_seed(self.seed)
        excel_file = self.dataset.affinity_data[0]
        pdb_file = self.dataset.pdb_path
        heavy_chain_id = self.dataset.heavy_chain
        light_chain_id = self.dataset.light_chain
        antigen_chains = self.dataset.antigen_chains
        chain_order = self.dataset.chain_order
        pdb_name_offset = self.dataset.pdb

        # load affinity data
        df = pd.read_csv(excel_file)

        for idx, row in tqdm(df.iterrows()):

            structure = load_structure(pdb_file)
            _, native_seqs = extract_coords_from_complex(structure)

            mutated_seqs = {}
            mutated_seqs[heavy_chain_id] = row['mut_heavy_chain_seq']
            mutated_seqs[light_chain_id] = native_seqs[light_chain_id]
            for c in antigen_chains:
                mutated_seqs[c] = native_seqs[c]

            mut_info = []
            for chain in chain_order:
                native_seq = native_seqs[chain]
                mut_seq = mutated_seqs[chain]
                mutations = [(i+1, native_seq[i], mut_seq[i]) for i in range(len(native_seq)) if native_seq[i] != mut_seq[i]]

                for single_mutation in mutations:
                    pos, wt, mt = single_mutation
                    mut_info.append(f'{wt}{chain}{pos}{mt}')

            mut_info_ = ','.join(mut_info)
            mut_info_ += ';'
            mutations = mut_info_[:-1]
            
            pdb_name = args.name + f'_{mutations}'
            _ = numbered_to_sequential(pdb_file, f'{dir_name}/{pdb_name}.pdb')
            mutated_path = mutate_pdb_sequence(f'{dir_name}/{pdb_name}.pdb', mut_info, f'{dir_name}/{pdb_name}.pdb')
            if not Path(mutated_path).exists():
                print(f'{mutated_path} not found')
            score = eval_mpnn_score(mutated_path, args.gpu, dir_name, args.mpnn_path)
            df.at[idx, 'log-likelihood'] = score
        return df

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numbered_to_sequential(input_pdb, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input_structure", input_pdb)
    
    mapping_dict = {}  # To store the mapping: (original_chain, original_resseq, original_icode) -> new_resseq

    new_structure = deepcopy(structure)    
    for model in new_structure:
        for chain in model:
            new_residue_id = 1 # Start renumbering residues from 1
            for residue in chain:
                # Get original residue info
                original_id = (chain.id, residue.id[0], residue.id[1], residue.id[2])
                residue.id = (residue.id[0], residue.id[1]+10000, residue.id[2])

                # Renumber residue
                new_id = (residue.id[0], new_residue_id, ' ')
                
                # residue.id = new_id  # Update only resseq

                # Store mapping
                mapping_dict[original_id] = new_id #new_residue_id
                new_residue_id += 1  # Increment residue ID sequentially

    for original_model, new_model in zip(structure, new_structure):
        for original_chain, new_chain in zip(original_model, new_model):
            for original_residue, new_residue in zip(original_chain, new_chain):
                original_id = (original_chain.id, original_residue.id[0], original_residue.id[1], original_residue.id[2])
                new_id = mapping_dict[original_id]
                new_residue.id = new_id

    # Write renumbered structure to a new file
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb)

    return mapping_dict

def mutate_pdb_sequence(pdb_path, mut_info, output_pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    for mutation in mut_info:
        original_residue = mutation[0]
        chain_id = mutation[1]
        residue_index = int(mutation[2:-1])
        mutated_residue = mutation[-1]


        chain = structure[0][chain_id]


        found = False
        for residue in chain:
            if residue.get_id()[1] == residue_index:
                resname = residue.get_resname().upper()[0] + residue.get_resname().lower()[1:]
                resname = three_to_one[resname]
                assert resname == original_residue, \
                    f"Original residue {original_residue} expected at chain {chain_id} residue {residue_index}, " \
                    f"but found {resname}"
                
                residue.resname = one_to_three[mutated_residue].upper()
                found = True
                break
        
        if not found:
            raise ValueError(f"Residue {residue_index} in chain {chain_id} not found or mismatch in the PDB file.")

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

    return output_pdb_path


def eval_mpnn_score(pdb_file, gpu, output_dir, MPNN_PATH):
    result = subprocess.run([
        "python", f"{MPNN_PATH}/protein_mpnn_run.py",
        "--pdb_path", pdb_file,
        "--out_folder", output_dir,
        "--score_only", "1",
        "--seed", "37",
        "--gpu", str(gpu),
    ], capture_output=True, text=True)

    if result.returncode == 0:  # Check if the command was successful
        filename = os.path.splitext(os.path.basename(pdb_file))[0]
        data = f'{output_dir}/score_only/{filename}_pdb.npz'  # Corrected variable name
        if os.path.exists(data):  # Check if the file exists
            # print(f'    extracting score from {data}')
            loaded_data = np.load(data)  # keys=['score', 'global_score', 'S', 'seq_str']
            score = loaded_data['score'][0]
            return -score  # Return the score
        else:
            print(f"Error: File does not exist: {data}")  # Print error if file is missing
            return None  # Return None if the file is missing
    else:
        print(f"Error: {result.stderr}")  # Print error if command fails
        return None  # Return None if there was an error
