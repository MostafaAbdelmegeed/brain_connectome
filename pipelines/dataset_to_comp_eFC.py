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
    parser.add_argument('--dataset', type=str, default='ADNI', help='Dataset to use.', choices=['ADNI', 'PPMI'])
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use.')
    args = parser.parse_args()
    return args

def compute_eTS(timeseries, device):
    N, T = timeseries.shape
    z_timeseries = (timeseries - timeseries.mean(dim=1, keepdim=True).to(device)) / timeseries.std(dim=1, keepdim=True).to(device)
    eTS = torch.einsum('it,jt->ijt', z_timeseries, z_timeseries)
    triu_indices = torch.triu_indices(N, N, offset=1)
    eTS = eTS[triu_indices[0], triu_indices[1]]
    return eTS

def compute_edge_features(eTS):
    return eTS

def compute_eFC(edge_features, device):
    eFC = torch.corrcoef(edge_features.to(device))
    return eFC

def pipeline(timeseries_data, device):
    M, N, T = timeseries_data.shape
    E = N * (N - 1) // 2
    eFC_matrices = torch.zeros((M, E, E), device=device)
    
    for i in tqdm(range(M), desc="Computing eFC matrices"):
        eTS = compute_eTS(timeseries_data[i], device)
        edge_features = compute_edge_features(eTS)
        eFC = compute_eFC(edge_features, device)
        eFC_matrices[i] = eFC
    
    return eFC_matrices

def compute_interhemispheric_eTS(left_timeseries, right_timeseries, device):
    eTS = torch.einsum('it,jt->ijt', left_timeseries, right_timeseries).to(device)
    return eTS

def compute_interhemispheric_eFC(eTS, device):
    L, R, T = eTS.shape
    eTS_reshaped = eTS.reshape(L * R, T).to(device)
    eFC = torch.corrcoef(eTS_reshaped)
    return eFC

def interhemispheric_pipeline(timeseries_data, left_indices, right_indices, device):
    M, N, T = timeseries_data.shape
    eFC_matrices = []

    for i in tqdm(range(M), desc="Computing interhemispheric eFC matrices"):
        left_timeseries = (timeseries_data[i, left_indices, :] - timeseries_data[i, left_indices, :].mean(dim=1, keepdim=True).to(device)) / timeseries_data[i, left_indices, :].std(dim=1, keepdim=True).to(device)
        right_timeseries = (timeseries_data[i, right_indices, :] - timeseries_data[i, right_indices, :].mean(dim=1, keepdim=True).to(device)) / timeseries_data[i, right_indices, :].std(dim=1, keepdim=True).to(device)

        eTS = compute_interhemispheric_eTS(left_timeseries, right_timeseries, device)
        eFC = compute_interhemispheric_eFC(eTS, device)
        eFC_matrices.append(eFC.cpu())
    
    eFC_matrices = torch.stack(eFC_matrices)
    return eFC_matrices

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    data = read_adni_timeseries('./data/ADNI') if args.dataset == 'ADNI' else read_ppmi_timeseries('./data/PPMI')
    labels = data['label']
    timeseries_data = data['timeseries']
    timeseries_data = torch.tensor(np.copy(timeseries_data), dtype=torch.float32).to(device)
    
    print(f'Timeseries data shape: {timeseries_data.shape}, Labels shape: {labels.shape}')
    
    left_indices = [i for i in range(0, timeseries_data.shape[1], 2)]
    right_indices = [i for i in range(1, timeseries_data.shape[1], 2)]
    
    intra_left = timeseries_data[:, left_indices, :]
    intra_right = timeseries_data[:, right_indices, :]
    homotopic = timeseries_data[:, left_indices, :] - timeseries_data[:, right_indices, :]
    
    eFC = pipeline(timeseries_data, device)
    inter_eFC = interhemispheric_pipeline(timeseries_data, left_indices, right_indices, device)
    intra_left_eFC = pipeline(intra_left, device)
    intra_right_eFC = pipeline(intra_right, device)
    homo_eFC = pipeline(homotopic, device)

    print(f'eFC Shape: {eFC.shape}')
    print(f'inter_eFC Shape: {inter_eFC.shape}')
    print(f'intra_left_eFC Shape: {intra_left_eFC.shape}')
    print(f'intra_right_eFC Shape: {intra_right_eFC.shape}')
    print(f'homotopic_eFC Shape: {homo_eFC.shape}')
    print(f'labels Shape: {labels.shape}')

    destination = './data/' + args.dataset + '_efc.pth'
    
    torch.save({
        'eFC': eFC,
        'intra_left': intra_left_eFC,
        'intra_right': intra_right_eFC,
        'inter': inter_eFC,
        'homo': homo_eFC,
        'label': labels
    }, destination)
    print(f'eFC matrices and labels saved to {destination}')
