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

def extract_hemisphere_timeseries(timeseries_data, indices):
    return timeseries_data[:, indices, :]

def pipeline(timeseries_data, indices):
    M, _, T = timeseries_data.shape
    hemisphere_data = timeseries_data[:, indices, :]
    E = len(indices) * (len(indices) - 1) // 2
    eFC_matrices = torch.zeros((M, E, E))

    for i in tqdm(range(M), desc="Computing eFC matrices"):
        eTS = compute_eTS(hemisphere_data[i])
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features)
        eFC_matrices[i] = eFC[:E, :E]  # Ensure the correct shape

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
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    
    # Compute eFC matrices for left and right hemispheres
    left_eFC_matrices = pipeline(timeseries_data, left_indices)
    right_eFC_matrices = pipeline(timeseries_data, right_indices)
    
    print(f'Left hemisphere eFC matrices shape: {left_eFC_matrices.shape}')
    print(f'Right hemisphere eFC matrices shape: {right_eFC_matrices.shape}')

    destination = './data/' + args.dataset + '_hemispheres_eFC.pth'
    
    torch.save({
        'left': left_eFC_matrices, 
        'right': right_eFC_matrices, 
        'labels': labels
    }, destination)
    print(f'Left and right hemisphere eFC matrices and labels saved to {destination}')

