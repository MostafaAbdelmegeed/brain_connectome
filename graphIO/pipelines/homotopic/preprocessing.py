import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold

def check_for_nans_infs(matrix, name="Matrix"):
    if np.isnan(matrix).any():
        raise ValueError(f"{name} contains NaNs.")
    if np.isinf(matrix).any():
        raise ValueError(f"{name} contains Infs.")

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

def calculate_inter_hemispheric_asymmetry_vector_aal116(matrix, method='abs_diff', epsilon=1e-10):
    means = np.mean(matrix, axis=1)
    ai = []
    for i in range(0, len(means)-1, 2):
        if method == 'abs_diff':
            ai.append(np.abs(means[i] - means[i+1]))
        elif method == 'ai':
            ai.append((means[i] - means[i+1]) / (means[i] + means[i+1] + epsilon))
        elif method == 'sq_diff':
            ai.append((means[i] - means[i+1])**2)
        elif method == 'log_ratio':
            ai.append(np.log((means[i] + epsilon) / (means[i+1] + epsilon)))
        elif method == 'perc_diff':
            ai.append((means[i] - means[i+1]) / ((means[i] + means[i+1] + epsilon) / 2) * 100)
        elif method == 'ratio':
            ai.append((means[i] + epsilon) / (means[i+1] + epsilon))
        elif method == 'smape':
            ai.append(100 * (np.abs(means[i] - means[i+1]) / ((np.abs(means[i]) + np.abs(means[i+1]) + epsilon) / 2)))
    return ai

def suppress_interhemispheric_connectivity(matrix):
    num_regions = matrix.shape[0] // 2
    for i in range(num_regions):
        left_region_index = 2 * i
        right_region_index = 2 * i + 1
        for j in range(num_regions):
            if j != i:
                left_region_other = 2 * j
                right_region_other = 2 * j + 1
                matrix[left_region_index, right_region_other] = 0
                matrix[right_region_index, left_region_other] = 0
    return matrix

def homotopic(i, j):
    return i//2 == j//2

def construct_graph(matrix, curvatures=[], asymmetry_indices=[], threshold=0.3, use_curvature=False, use_asym=False):
    G = nx.Graph()
    num_nodes = matrix.shape[0]
    for i in range(num_nodes):
        hemisphere = 0 if i % 2 == 0 else 1
        G.add_node(i, strength=matrix[i, :].mean(), hemisphere=hemisphere)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(matrix[i, j]) > threshold:
                interhemispheric = 1 if (i % 2 != j % 2) else 0
                curvature = curvatures[i][j] if use_curvature else 0.0
                G.add_edge(i, j, weight=matrix[i, j], interhemispheric=interhemispheric, curvature=curvature, asymmetry_index=asymmetry_indices[i//2] if homotopic(i,j) and use_asym else 0.0)
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree[node]
    clustering_coeffs = nx.clustering(G)
    for node, coeff in clustering_coeffs.items():
        G.nodes[node]['clustering'] = coeff
    betweenness = nx.betweenness_centrality(G)
    for node, centrality in betweenness.items():
        G.nodes[node]['betweenness'] = centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    for u, v, data in G.edges(data=True):
        data['edge_betweenness'] = edge_betweenness[(u, v)]
    max_weight = max([data['weight'] for _, _, data in G.edges(data=True)])
    for u, v, data in G.edges(data=True):
        data['normalized_weight'] = data['weight'] / max_weight
    return G

def convert_to_pyg_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([[G[u][v]['weight'], G[u][v]['interhemispheric'], G[u][v]['edge_betweenness'], G[u][v]['normalized_weight'], G[u][v]['curvature'], G[u][v]['asymmetry_index']] for u, v in G.edges], dtype=torch.float)
    node_features = np.array([[G.nodes[i]['strength'], G.nodes[i]['hemisphere'], G.nodes[i]['degree'], G.nodes[i]['clustering'], G.nodes[i]['betweenness']] for i in G.nodes])
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def load_data(dataset_path, suppress_ihc=False):
    ppmi_dataset = torch.load(dataset_path)
    connectivity_matrices = center_matrices(ppmi_dataset['data'].numpy())
    connectivity_labels = ppmi_dataset['class_label'].numpy().reshape(-1, 1)
    curvatures = center_matrices(ppmi_dataset['curvatures'].numpy())
    asymmetry_indices = center_matrices(np.array([calculate_inter_hemispheric_asymmetry_vector_aal116(matrix) for matrix in tqdm(connectivity_matrices, desc='Computing asymmetry indices')]))

    # Check for NaNs and Infs
    check_for_nans_infs(connectivity_matrices, "Connectivity Matrices")
    check_for_nans_infs(curvatures, "Curvatures")
    check_for_nans_infs(asymmetry_indices, "Asymmetry Indices")

    if suppress_ihc:
        connectivity_matrices = np.array([suppress_interhemispheric_connectivity(matrix) for matrix in connectivity_matrices])
        print(f'{len(connectivity_matrices)} matrices got non-homotopic interhemispherical connections suppression')

    return connectivity_matrices, connectivity_labels, curvatures, asymmetry_indices

def create_graphs(connectivity_matrices, curvatures, asymmetry_indices, suppress_threshold, use_curvature=False, use_asym=False):
    graphs = []
    for i in tqdm(range(len(connectivity_matrices)), desc='Creating graphs'):
        graphs.append(construct_graph(connectivity_matrices[i], curvatures[i], asymmetry_indices[i//2], suppress_threshold, use_curvature, use_asym))
    return graphs

class BrainConnectivityDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.y = self.labels[idx]
        return data

def create_data_loaders(graphs, labels, n_splits=5, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    loaders = []
    for train_index, test_index in skf.split(graphs, labels):
        train_graphs = [graphs[i] for i in train_index]
        train_labels = labels[train_index]
        test_graphs = [graphs[i] for i in test_index]
        test_labels = labels[test_index]
        train_dataset = BrainConnectivityDataset(train_graphs, train_labels)
        test_dataset = BrainConnectivityDataset(test_graphs, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loaders.append((train_loader, test_loader))
    return loaders
