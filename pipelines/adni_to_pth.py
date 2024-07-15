import argparse
import sys
from pathlib import Path
# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_adj_matrices_from_directory
from graphIO.preprocess import standardize_matrices
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import pandas as pd

# python pipelines/convert_adni_to_pyg_pth.py --source "C:/Users/mosta/OneDrive - UNCG/Academics/CSC 699 - Thesis/data/ADNI/AAL90_FC" --labels "C:/Users/mosta/OneDrive - UNCG/Academics/CSC 699 - Thesis/data/ADNI/label-2cls_new.csv" --destination "data/adni.pth"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Directory containing the ADNI data.')
    parser.add_argument('--labels', type=str, required=True, help='Path to the labels file.')
    parser.add_argument('--method', type=str, default='correlation', help='Use correlation matrices', choices=['correlation', 'curvature'])
    parser.add_argument('--destination', type=str, default='adni.pth', help='Save path for the processed data.')
    args = parser.parse_args()
    return args

def construct_graph(matrix, threshold=0.0):
    G = nx.Graph()
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=matrix[i, j])
    return G

def read_labels(path):
    df = pd.read_csv(path)
    keys = list(df['subject_id'].values)
    labels = list(df['Label1'].values)
    return dict(zip(keys, labels))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python adni_to_pth.py <directory>")
        sys.exit(1)
    args = parse_args()
    source = args.source
    destination = args.destination
    method = args.method
    data = read_adj_matrices_from_directory(source, as_tensor=True)
    labels_dict = read_labels(args.labels)
    connectivity_matrices = []
    labels = []
    for key in data.keys():
        connectivity_matrices.append(data[key])
        labels.append(labels_dict[key.split('-')[1].split('_')[0]])
    connectivity_matrices = torch.stack(connectivity_matrices)
    print(f'Connectivity matrices: {len(connectivity_matrices)}, Labels: {len(labels)}')
    labels = np.array(labels)
    print(f'Connectivity matrices shape: {connectivity_matrices.shape}, Labels shape: {labels.shape}')
    print(f'Connectivity matrices dtype: {connectivity_matrices.dtype}, Labels dtype: {labels.dtype}')
    print(f'Connectivity matrices min: {connectivity_matrices.min()}, Labels min: {labels.min()}')
    print(f'Connectivity matrices max: {connectivity_matrices.max()}, Labels max: {labels.max()}')
    torch.save({'matrix': connectivity_matrices, 'label': torch.tensor(labels, dtype=torch.int64)}, destination)
    print(f'Saved data to {destination}')