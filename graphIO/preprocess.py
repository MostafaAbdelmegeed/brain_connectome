import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from .math import *


def center_matrices(matrices):
    return standardize_matrices(normalize_matrices(matrices))

def normalize_matrices(matrices):
    min_val = np.min(matrices)
    max_val = np.max(matrices)
    normalized = (matrices - min_val) / (max_val - min_val)
    return normalized

def standardize_matrices(matrices):
    mean_val = np.mean(matrices)
    std_dev = np.std(matrices)
    standardized = (matrices - mean_val) / std_dev
    return standardized


################# AAL116 
def construct_graph(matrix, threshold=0.3):
    G = nx.Graph()
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        G.add_node(i)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=matrix[i, j])
    return G

def convert_to_pyg_data(G, num_node_features):
    # Convert edge list to tensor
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    
    # Convert edge attributes to tensor
    edge_attr = torch.tensor([[G[u][v]['weight']] for u, v in G.edges], dtype=torch.float).view(-1, 1)

    # Create node features based on indices
    num_nodes = G.number_of_nodes()
    node_features = np.eye(num_nodes, num_node_features)
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create PyTorch Geometric data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def create_graphs(connectivity_matrices):
    graphs = []
    for i in tqdm(range(len(connectivity_matrices)), desc='Creating graphs'):
        graphs.append(construct_graph(connectivity_matrices[i]))
    return graphs

################# AAL116 
