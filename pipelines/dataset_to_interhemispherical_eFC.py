import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse
import os
import tempfile

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_adni_timeseries, read_ppmi_timeseries

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ADNI', 'PPMI'], help='Dataset to use.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas to use.')
    parser.add_argument('--method', type=str, default='timeseries', help='Use correlation matrices', choices=['correlation', 'curvature', 'timeseries'])
    parser.add_argument('--destination', type=str, default='interhemispherical_eFC_matrices.pth', help='Save path for the processed data.')
    args = parser.parse_args()
    return args

def compute_eTS(timeseries):
    N, T = timeseries.shape
    z_timeseries = (timeseries - timeseries.mean(dim=1, keepdim=True)) / timeseries.std(dim=1, keepdim=True)
    eTS = torch.einsum('it,jt->ijt', z_timeseries, z_timeseries)
    triu_indices = torch.triu_indices(N, N, offset=1)
    eTS = eTS[triu_indices[0], triu_indices[1]]
    return eTS

def compute_edge_features(eTS):
    return eTS

def compute_eFC(edge_features):
    num_edges = edge_features.shape[0]
    eFC = torch.zeros((num_edges, num_edges), dtype=torch.float32)

    for i in range(num_edges):
        for j in range(i, num_edges):
            eFC[i, j] = torch.corrcoef(edge_features[i], edge_features[j])[0, 1]
            eFC[j, i] = eFC[i, j]
    
    return eFC

def identify_interhemispherical_pairs(left_indices, right_indices):
    interhemispherical_pairs = [(left, right) for left in left_indices for right in right_indices]
    return interhemispherical_pairs

def extract_timeseries_for_pairs(timeseries_data, pairs):
    indices = [index for pair in pairs for index in pair]
    return timeseries_data[:, indices, :]

def interhemispherical_pipeline(timeseries_data, pairs, temp_dir):
    M, _, T = timeseries_data.shape
    pairs_data = extract_timeseries_for_pairs(timeseries_data, pairs)
    E = len(pairs)

    temp_files = []

    for i in tqdm(range(M), desc="Computing interhemispherical eFC matrices"):
        eTS = compute_eTS(pairs_data[i])
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features)

        temp_file = os.path.join(temp_dir, f'eFC_matrix_{i}.pt')
        torch.save(eFC[:E, :E], temp_file)
        temp_files.append(temp_file)

    return temp_files

if __name__ == "__main__":
    args = parse_args()

    data = read_adni_timeseries('./data/ADNI') if args.dataset == 'ADNI' else read_ppmi_timeseries('./data/PPMI')
    labels = data['label']
    timeseries_data = data['timeseries']

    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    # Create lists of indices for left and right hemisphere regions
    left_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 == 0]
    right_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 != 0]
    
    # Identify interhemispherical pairs
    interhemispherical_pairs = identify_interhemispherical_pairs(left_indices, right_indices)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')

    # Create a temporary directory to store intermediate results
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = interhemispherical_pipeline(timeseries_data, interhemispherical_pairs, temp_dir)
        
        # Load the results from the temporary files and combine them
        eFC_matrices = []
        for temp_file in temp_files:
            eFC_matrices.append(torch.load(temp_file))
        eFC_matrices = torch.stack(eFC_matrices)
    
        print(f'Interhemispherical eFC matrices shape: {eFC_matrices.shape}')
    
        destination = './data/' + args.dataset + '_interhemispherical_eFC.pth'
        
        torch.save({
            'interhemispherical_eFC_matrices': eFC_matrices, 
            'labels': labels
        }, destination)
        print(f'Interhemispherical eFC matrices and labels saved to {destination}')
