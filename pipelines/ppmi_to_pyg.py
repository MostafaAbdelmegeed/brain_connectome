import argparse
import sys
from pathlib import Path
# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.io import read_ppmi_data_as_tensors
from graphIO.preprocess import standardize_matrices
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Directory containing the PPMI data.')
    parser.add_argument('--method', type=str, default='correlation', help='Method used to compute the adjacency matrices.')
    parser.add_argument('--atlas', type=str, default='AAL116', help='Atlas used to compute the adjacency matrices.')
    parser.add_argument('--destination', type=str, default='ppmi.pth', help='Save path for the processed data.')
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

def convert_to_pyg_data(G, num_node_features, label):
    # Convert edge list to tensor
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    # Convert edge attributes to tensor
    edge_attr = torch.tensor([[G[u][v]['weight']] for u, v in G.edges], dtype=torch.float).view(-1)
    # Create node features based on indices
    num_nodes = G.number_of_nodes()
    node_features = np.ones((num_nodes, num_nodes))
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


# python pipelines/convert_ppmi_to_pyg_pth.py --source "C:/Users/mosta/OneDrive - UNCG/Academics/CSC 699 - Thesis/data/ppmi" --destination data/ppmi_raw.pth


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_ppmi_to_pth.py <directory>")
        sys.exit(1)
    args = parse_args()
    source = args.source
    destination = args.destination
    atlas = args.atlas
    method = args.method
    data = read_ppmi_data_as_tensors(source, atlas=atlas, method=method)
    connectivity_matrices = standardize_matrices(data['matrix'].numpy())
    labels = data['label'].numpy()
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
    



    # torch.save(data, destination)