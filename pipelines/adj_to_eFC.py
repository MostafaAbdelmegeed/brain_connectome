import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_adni_timeseries, read_ppmi_timeseries

# python pipelines/convert_adni_to_eFC_pth.py --source "C:/Users/mosta/OneDrive - UNCG/Academics/CSC 699 - Thesis/data/ADNI" --destination "./data/adni_efc.pth"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Directory containing the ADNI data.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas to use.')
    parser.add_argument('--method', type=str, default='timeseries', help='Use correlation matrices', choices=['correlation', 'curvature', 'timeseries'])
    parser.add_argument('--destination', type=str, default='adni_efc.pth', help='Save path for the processed data.')
    parser.add_argument('--dataset', type=str, default='ADNI', help='Dataset to use.', choices=['ADNI', 'PPMI'])
    args = parser.parse_args()
    return args

def compute_eTS(timeseries):
    """
    Compute edge time series (eTS) for a given pair of regions.
    Args:
    - timeseries: Tensor of shape (N, T), where N is the number of regions and T is the number of time points.
    
    Returns:
    - eTS: Tensor of shape (N * (N - 1) // 2, T), representing the edge time series.
    """
    N, T = timeseries.shape
    z_timeseries = (timeseries - timeseries.mean(dim=1, keepdim=True)) / timeseries.std(dim=1, keepdim=True)
    eTS = torch.einsum('it,jt->ijt', z_timeseries, z_timeseries)
    triu_indices = torch.triu_indices(N, N, offset=1)
    eTS = eTS[triu_indices[0], triu_indices[1]]
    return eTS

def compute_edge_features(eTS):
    """
    Compute edge features from edge time series.
    Args:
    - eTS: Tensor of shape (N * (N - 1) // 2, T), representing the edge time series.
    
    Returns:
    - edge_features: Tensor of shape (E, T), where E = N * (N - 1) // 2, representing the edge features.
    """
    return eTS

def compute_eFC(edge_features):
    """
    Compute edge functional connectivity (eFC) from edge features.
    Args:
    - edge_features: Tensor of shape (E, T), where E = N * (N - 1) // 2, representing the edge features.
    
    Returns:
    - eFC: Tensor of shape (E, E), representing the edge functional connectivity.
    """
    eFC = torch.corrcoef(edge_features)  # Shape: (6670, 6670)
    return eFC

def pipeline(timeseries_data):
    """
    Pipeline to compute edge functional connectivity (eFC) matrices from timeseries data.
    Args:
    - timeseries_data: Tensor of shape (M, N, T), where M is the number of records, N is the number of regions, and T is the number of time points.
    
    Returns:
    - eFC_matrices: Tensor of shape (M, E, E), representing the edge functional connectivity matrices for each record.
    """
    M, N, T = timeseries_data.shape
    E = N * (N - 1) // 2  # Number of edges for an undirected graph
    eFC_matrices = torch.zeros((M, E, E))
    
    for i in tqdm(range(M), desc="Computing eFC matrices"):
        eTS = compute_eTS(timeseries_data[i])
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features)
        eFC_matrices[i] = eFC
    
    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()
    source = args.source
    destination = args.destination
    method = args.method
    atlas = args.atlas

    data = read_adni_timeseries(source) if args.dataset == 'ADNI' else read_ppmi_timeseries(source)
    labels = data['label']
    timeseries_data = data['timeseries']
    
    # Ensure the NumPy array is writable and convert to PyTorch tensor
    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    eFC_matrices = pipeline(timeseries_data)
    print(f'eFC matrices shape: {eFC_matrices.shape}')
    
    # Save the eFC matrices and labels
    torch.save({'eFC_matrices': eFC_matrices, 'labels': labels}, destination)
    print(f'eFC matrices and labels saved to {destination}')
