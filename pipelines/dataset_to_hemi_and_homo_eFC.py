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
    parser.add_argument('--destination', type=str, default='efc_matrices.pth', help='Save path for the processed data.')
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
    """
    Identify homotopic pairs of regions.
    Args:
    - N: Total number of regions (should be even).
    
    Returns:
    - homotopic_pairs: List of tuples representing homotopic pairs.
    """
    homotopic_pairs = [(i, i + 1) for i in range(0, N, 2)]
    return homotopic_pairs

def extract_hemisphere_timeseries(timeseries_data, indices):
    """
    Extract time series data for a specific hemisphere.
    Args:
    - timeseries_data: Tensor of shape (M, N, T)
    - indices: List of indices for the hemisphere
    
    Returns:
    - hemisphere_timeseries: Tensor of shape (M, len(indices), T)
    """
    return timeseries_data[:, indices, :]

def extract_homotopic_timeseries(timeseries_data, homotopic_pairs):
    """
    Extract time series data for homotopic pairs.
    Args:
    - timeseries_data: Tensor of shape (M, N, T)
    - homotopic_pairs: List of tuples representing homotopic pairs
    
    Returns:
    - homotopic_timeseries: Tensor of shape (M, len(homotopic_pairs) * 2, T)
    """
    indices = [index for pair in homotopic_pairs for index in pair]
    return timeseries_data[:, indices, :]

def pipeline(timeseries_data, indices):
    M, N, T = timeseries_data.shape
    hemisphere_data = extract_hemisphere_timeseries(timeseries_data, indices)
    E = len(indices) * (len(indices) - 1) // 2
    eFC_matrices = torch.zeros((M, E, E))
    
    for i in tqdm(range(M), desc="Computing eFC matrices"):
        eTS = compute_eTS(hemisphere_data[i])
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features)
        eFC_matrices[i] = eFC
    
    return eFC_matrices

def homotopic_pipeline(timeseries_data, homotopic_pairs):
    M, N, T = timeseries_data.shape
    homotopic_timeseries = extract_homotopic_timeseries(timeseries_data, homotopic_pairs)
    E = len(homotopic_pairs) * (len(homotopic_pairs) - 1) // 2
    eFC_matrices = torch.zeros((M, E, E))
    
    for i in tqdm(range(M), desc="Computing homotopic eFC matrices"):
        eTS = compute_eTS(homotopic_timeseries[i])
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features)
        eFC_matrices[i] = eFC
    
    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()
    destination = args.destination

    data = read_adni_timeseries('./data/ADNI') if args.dataset == 'ADNI' else read_ppmi_timeseries('./data/PPMI')
    labels = data['label']
    timeseries_data = data['timeseries']

    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    # Create lists of indices for left and right hemisphere regions
    left_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 == 0]
    right_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 != 0]
    
    # Identify homotopic pairs
    homotopic_pairs = identify_homotopic_pairs(timeseries_data.shape[1])
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    
    left_eFC_matrices = pipeline(timeseries_data, left_indices)
    right_eFC_matrices = pipeline(timeseries_data, right_indices)
    homotopic_eFC_matrices = homotopic_pipeline(timeseries_data, homotopic_pairs)
    
    print(f'Left hemisphere eFC matrices shape: {left_eFC_matrices.shape}')
    print(f'Right hemisphere eFC matrices shape: {right_eFC_matrices.shape}')
    print(f'Homotopic eFC matrices shape: {homotopic_eFC_matrices.shape}')
    
    torch.save({
        'left_eFC_matrices': left_eFC_matrices, 
        'right_eFC_matrices': right_eFC_matrices, 
        'homotopic_eFC_matrices': homotopic_eFC_matrices, 
        'labels': labels
    }, destination)
    print(f'eFC matrices and labels saved to {destination}')
