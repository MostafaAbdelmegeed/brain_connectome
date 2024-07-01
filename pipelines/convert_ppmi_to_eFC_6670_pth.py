import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_ppmi_timeseries

# python pipelines/convert_ppmi_to_eFC_pth.py --source "./data/ppmi" --destination "./data/ppmi_efc.pth" --gpu 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Directory containing the PPMI data.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas to use.')
    parser.add_argument('--method', type=str, default='timeseries', help='Use correlation matrices', choices=['correlation', 'curvature', 'timeseries'])
    parser.add_argument('--destination', type=str, default='ppmi_efc.pth', help='Save path for the processed data.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    args = parser.parse_args()
    return args

def compute_eTS(timeseries):
    """
    Compute edge time series (eTS) for a given pair of regions.
    Args:
    - timeseries: Tensor of shape (N, T), where N is the number of regions and T is the number of time points.
    
    Returns:
    - eTS: Tensor of shape (N, N, T), representing the edge time series.
    """
    N, T = timeseries.shape
    z_timeseries = (timeseries - timeseries.mean(dim=1, keepdim=True)) / timeseries.std(dim=1, keepdim=True)
    eTS = torch.einsum('it,jt->ijt', z_timeseries, z_timeseries)
    return eTS

def compute_edge_features(eTS):
    """
    Compute edge features from edge time series.
    Args:
    - eTS: Tensor of shape (N, N, T), representing the edge time series.
    
    Returns:
    - edge_features: Tensor of shape (E, T), where E = N * N, representing the edge features.
    """
    N, _, T = eTS.shape
    edge_features = eTS.reshape(N * N, T)  # Shape: (6670, T)
    return edge_features

def compute_eFC(edge_features):
    """
    Compute edge functional connectivity (eFC) from edge features.
    Args:
    - edge_features: Tensor of shape (E, T), where E = N * N, representing the edge features.
    
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
    E = N * N  # Number of edges
    eFC_matrices = torch.zeros((M, E, E), device=timeseries_data.device)
    
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
    gpu_id = args.gpu

    # Set the device based on the GPU ID
    device = torch.device('cpu')

    data = read_ppmi_timeseries(source)
    labels = data['label']
    timeseries_data = data['timeseries']
    
    # Ensure the NumPy array is writable and convert to PyTorch tensor
    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    # Move data to the selected GPU if available
    timeseries_data = timeseries_data.to(device)
    labels = labels.to(device)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    eFC_matrices = pipeline(timeseries_data)
    print(f'eFC matrices shape: {eFC_matrices.shape}')
    
    # Save the eFC matrices and labels
    torch.save({'eFC_matrices': eFC_matrices, 'labels': labels}, destination)
    print(f'eFC matrices and labels saved to {destination}')
