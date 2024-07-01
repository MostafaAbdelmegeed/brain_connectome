import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import argparse

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_adni_timeseries

# python pipelines/convert_adni_to_eFC_pth.py --source "C:\Users\mosta\OneDrive - UNCG\Academics\CSC 699 - Thesis\data\ADNI" --destination "./data/adni_efc.pth"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Directory containing the ADNI data.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas to use.')
    parser.add_argument('--method', type=str, default='timeseries', help='Use correlation matrices', choices=['correlation', 'curvature', 'timeseries'])
    parser.add_argument('--destination', type=str, default='adni_efc.pth', help='Save path for the processed data.')
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

def compute_eFC(eTS):
    """
    Compute edge functional connectivity (eFC) from edge time series (eTS).
    Args:
    - eTS: Tensor of shape (N, N, T), representing the edge time series.
    
    Returns:
    - eFC: Tensor of shape (N, N), representing the edge functional connectivity.
    """
    N, _, T = eTS.shape
    eTS_reshaped = eTS.reshape(N * N, T)
    eFC = torch.corrcoef(eTS_reshaped)
    return eFC[:N, :N]

def pipeline(timeseries_data):
    """
    Pipeline to compute edge functional connectivity (eFC) matrices from timeseries data.
    Args:
    - timeseries_data: Tensor of shape (M, N, T), where M is the number of records, N is the number of regions, and T is the number of time points.
    
    Returns:
    - eFC_matrices: Tensor of shape (M, N, N), representing the edge functional connectivity matrices for each record.
    """
    M, N, T = timeseries_data.shape
    eFC_matrices = torch.zeros((M, N, N), device=timeseries_data.device)
    
    for i in tqdm(range(M), desc="Computing eFC matrices"):
        eTS = compute_eTS(timeseries_data[i])
        eFC_matrices[i] = compute_eFC(eTS)
    
    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()
    source = args.source
    destination = args.destination
    method = args.method
    atlas = args.atlas
    data = read_adni_timeseries(source)
    labels = data['label']
    timeseries_data = data['timeseries']
    
    # Ensure the NumPy array is writable and convert to PyTorch tensor
    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32)
    
    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timeseries_data = timeseries_data.to(device)
    labels = labels.to(device)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    eFC_matrices = pipeline(timeseries_data)
    print(f'eFC matrices shape: {eFC_matrices.shape}')
    
    # Save the eFC matrices and labels
    torch.save({'eFC_matrices': eFC_matrices, 'labels': labels}, destination)
    print(f'eFC matrices and labels saved to {destination}')
