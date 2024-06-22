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


def convert_to_pyg_data(G, num_of_node_features, label):
    # Convert edge list to tensor
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    # Convert edge attributes to tensor
    edge_attr = torch.tensor([[G[u][v]['weight']] for u, v in G.edges], dtype=torch.float).view(-1)
    # Create node features based on indices
    num_nodes = G.number_of_nodes()
    node_features = np.ones((num_nodes, num_of_node_features))
    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long)
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return data

def create_graphs(connectivity_matrices):
    graphs = []
    for i in tqdm(range(len(connectivity_matrices)), desc='Creating graphs'):
        graphs.append(construct_graph(connectivity_matrices[i]))
    return graphs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_ppmi_to_pth.py <directory>")
        sys.exit(1)
    args = parse_args()
    source = args.source
    destination = args.destination
    method = args.method
    data = read_adj_matrices_from_directory(source)
    labels_dict = read_labels(args.labels)
    connectivity_matrices = []
    labels = []
    for key in data.keys():
        connectivity_matrices.append(data[key])
        labels.append(labels_dict[key.split('-')[1].split('_')[0]])
    connectivity_matrices = standardize_matrices(np.array(connectivity_matrices))
    labels = np.array(labels)
    print(f'Connectivity matrices shape: {connectivity_matrices.shape}, Labels shape: {labels.shape}')
    print(f'Connectivity matrices dtype: {connectivity_matrices.dtype}, Labels dtype: {labels.dtype}')
    print(f'Connectivity matrices min: {connectivity_matrices.min()}, Labels min: {labels.min()}')
    print(f'Connectivity matrices max: {connectivity_matrices.max()}, Labels max: {labels.max()}')
    print(f'Connectivity matrices mean: {connectivity_matrices.mean()}, Labels mean: {labels.mean()}')
    print(f'Connectivity matrices std: {connectivity_matrices.std()}, Labels std: {labels.std()}')
    graphs = create_graphs(connectivity_matrices)
    print(f'Number of graphs: {len(graphs)}')
    pyg_data = [convert_to_pyg_data(graphs[i], connectivity_matrices.shape[1], labels[i]) for i in range(len(graphs))]
    torch.save(pyg_data, destination)
    print(f'Saved PyTorch Geometric data to {destination}')