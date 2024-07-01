import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_adni_timeseries, read_ppmi_timeseries

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['ADNI', 'PPMI'], help='Dataset to use.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas to use.')
    parser.add_argument('--method', type=str, default='timeseries', help='Use correlation matrices', choices=['correlation', 'curvature', 'timeseries'])
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing.')
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
    eFC = torch.corrcoef(edge_features)
    return eFC

def identify_homotopic_pairs(N):
    homotopic_pairs = [(i, i + 1) for i in range(0, N, 2)]
    return homotopic_pairs

def identify_interhemispherical_pairs(left_indices, right_indices):
    interhemispherical_pairs = [(left, right) for left in left_indices for right in right_indices]
    return interhemispherical_pairs

def extract_timeseries_for_pairs(timeseries_data, pairs):
    indices = [index for pair in pairs for index in pair]
    return timeseries_data[:, indices, :]

def pipeline(timeseries_data, indices, batch_size):
    M, _, T = timeseries_data.shape
    hemisphere_data = timeseries_data[:, indices, :]
    E = len(indices) * (len(indices) - 1) // 2
    eFC_matrices = torch.zeros((M, E, E))

    for i in tqdm(range(0, M, batch_size), desc="Computing eFC matrices"):
        batch_end = min(i + batch_size, M)
        batch_data = hemisphere_data[i:batch_end]
        for j in range(batch_end - i):
            eTS = compute_eTS(batch_data[j])
            edge_features = compute_edge_features(eTS)
            eFC = compute_eFC(edge_features)
            eFC_matrices[i + j] = eFC[:E, :E]  # Ensure the correct shape

    return eFC_matrices

def pairs_pipeline(timeseries_data, pairs, batch_size):
    M, _, T = timeseries_data.shape
    pairs_data = extract_timeseries_for_pairs(timeseries_data, pairs)
    E = len(pairs)
    eFC_matrices = torch.zeros((M, E, E))

    for i in tqdm(range(0, M, batch_size), desc="Computing pairs eFC matrices"):
        batch_end = min(i + batch_size, M)
        batch_data = pairs_data[i:batch_end]
        for j in range(batch_end - i):
            eTS = compute_eTS(batch_data[j])
            edge_features = compute_edge_features(eTS)
            eFC = compute_eFC(edge_features)
            eFC_matrices[i + j] = eFC[:E, :E]  # Ensure the correct shape

    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()

    data = read_adni_timeseries('./data/ADNI') if args.dataset == 'ADNI' else read_ppmi_timeseries('./data/PPMI')
    labels = data['label']
    timeseries_data = data['timeseries']

    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    # Create lists of indices for left and right hemisphere regions
    left_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 == 0]
    right_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 != 0]
    
    # Identify homotopic pairs
    homotopic_pairs = identify_homotopic_pairs(timeseries_data.shape[1])
    
    # Identify interhemispherical pairs
    interhemispherical_pairs = identify_interhemispherical_pairs(left_indices, right_indices)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    
    left_eFC_matrices = pipeline(timeseries_data, left_indices, args.batch_size)
    right_eFC_matrices = pipeline(timeseries_data, right_indices, args.batch_size)
    homotopic_eFC_matrices = pairs_pipeline(timeseries_data, homotopic_pairs, args.batch_size)
    interhemispherical_eFC_matrices = pairs_pipeline(timeseries_data, interhemispherical_pairs, args.batch_size)
    
    print(f'Left hemisphere eFC matrices shape: {left_eFC_matrices.shape}')
    print(f'Right hemisphere eFC matrices shape: {right_eFC_matrices.shape}')
    print(f'Homotopic eFC matrices shape: {homotopic_eFC_matrices.shape}')
    print(f'Interhemispherical eFC matrices shape: {interhemispherical_eFC_matrices.shape}')

    destination = './data/' + args.dataset + '_' + 'parts_eFC.py'
    
    torch.save({
        'left_eFC_matrices': left_eFC_matrices, 
        'right_eFC_matrices': right_eFC_matrices, 
        'homotopic_eFC_matrices': homotopic_eFC_matrices,
        'interhemispherical_eFC_matrices': interhemispherical_eFC_matrices, 
        'labels': labels
    }, destination)
    print(f'eFC matrices and labels saved to {destination}')
