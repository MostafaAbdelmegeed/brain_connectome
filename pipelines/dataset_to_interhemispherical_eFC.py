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
    parser.add_argument('--destination', type=str, default='interhemispherical_eFC_matrices.pth', help='Save path for the processed data.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    args = parser.parse_args()
    return args

def compute_eTS(timeseries, device):
    N, T = timeseries.shape
    z_timeseries = (timeseries - timeseries.mean(dim=1, keepdim=True).to(device)) / timeseries.std(dim=1, keepdim=True).to(device)
    eTS = torch.einsum('it,jt->ijt', z_timeseries, z_timeseries).to(device)
    triu_indices = torch.triu_indices(N, N, offset=1).to(device)
    eTS = eTS[triu_indices[0], triu_indices[1]]
    return eTS

def compute_edge_features(eTS, device):
    return eTS.to(device)

def compute_eFC(edge_features, device):
    num_edges = edge_features.shape[0]
    eFC = torch.zeros((num_edges, num_edges), dtype=torch.float32, device=device)

    for i in range(num_edges):
        for j in range(i, num_edges):
            eFC[i, j] = torch.corrcoef(edge_features[i].to(device), edge_features[j].to(device))[0, 1]
            eFC[j, i] = eFC[i, j]
    
    return eFC

def identify_interhemispherical_pairs(left_indices, right_indices):
    interhemispherical_pairs = [(left, right) for left in left_indices for right in right_indices]
    return interhemispherical_pairs

def extract_timeseries_for_pairs(timeseries_data, pairs, device):
    indices = [index for pair in pairs for index in pair]
    return timeseries_data[:, indices, :].to(device)

def interhemispherical_pipeline(timeseries_data, pairs, device):
    M, _, T = timeseries_data.shape
    pairs_data = extract_timeseries_for_pairs(timeseries_data, pairs, device)
    E = len(pairs)
    eFC_matrices = torch.zeros((M, E, E), dtype=torch.float32, device=device)

    for i in tqdm(range(M), desc="Computing interhemispherical eFC matrices"):
        eTS = compute_eTS(pairs_data[i], device)
        edge_features = compute_edge_features(eTS, device)
        eFC = compute_eFC(edge_features, device)
        eFC_matrices[i] = eFC[:E, :E]  # Ensure the correct shape

    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    data = read_adni_timeseries('./data/ADNI') if args.dataset == 'ADNI' else read_ppmi_timeseries('./data/PPMI')
    labels = data['label']
    timeseries_data = data['timeseries']

    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32).to(device)
    
    # Create lists of indices for left and right hemisphere regions
    left_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 == 0]
    right_indices = [i for i in range(timeseries_data.shape[1]) if i % 2 != 0]
    
    # Identify interhemispherical pairs
    interhemispherical_pairs = identify_interhemispherical_pairs(left_indices, right_indices)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    
    interhemispherical_eFC_matrices = interhemispherical_pipeline(timeseries_data, interhemispherical_pairs, device)
    
    print(f'Interhemispherical eFC matrices shape: {interhemispherical_eFC_matrices.shape}')

    destination = './data/' + args.dataset + '_interhemispherical_eFC.pth'
    
    torch.save({
        'interhemispherical_eFC_matrices': interhemispherical_eFC_matrices, 
        'labels': labels
    }, destination)
    print(f'Interhemispherical eFC matrices and labels saved to {destination}')
