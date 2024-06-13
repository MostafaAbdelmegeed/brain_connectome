#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from nilearn import datasets, plotting
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from multi_head_gat_with_edge_features import GATWithEdgeFeatures
import torch.nn as nn
from tqdm import tqdm
import time
import argparse


# Argument parsing
parser = argparse.ArgumentParser(description='Brain Connectivity Analysis with GAT')
parser.add_argument('--suppress_threshold', type=float, default=0.3, help='Threshold to suppress interhemispheric connectivity')
parser.add_argument('--attention_heads', type=int, default=4, help='Number of attention heads')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--drop_out', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for leaky ReLU')
parser.add_argument('--layers', type=int, default=2, help='Number of layers in the model')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden features')
parser.add_argument('--out_features', type=int, default=8, help='Number of output features')
parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
parser.add_argument('--pth_dir', type=str, default='data/ppmi_w_curv.pth', help='Dataset Directory')
parser.add_argument('--suppress_ihc', action='store_true', help='Suppress all non-homotopic interhemispherical edges')
parser.add_argument('--use_asym', action='store_true', help='Uses Asymmetry index for homotopic edges')
parser.add_argument('--use_curvature', action='store_true', help='Uses curvature as additional features for edges')
parser.add_argument('--center_matrices', action='store_true', help='Center matrices')
parser.add_argument('--normalize_matrices', action='store_true', help='Normalize matrices')
parser.add_argument('--standardize_matrices', action='store_true', help='Standardize matrices')

args = parser.parse_args()

# CONSTANTS
DATASET_PATH = args.pth_dir
STUDY_CONNECTOME_INDEX = 20
NODE_FEATURES = 5
EDGE_FEATURES = 6

# HYPERPARAMETERS
SUPPRESS_THRESHOLD = args.suppress_threshold
ATTENTION_HEADS = args.attention_heads
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
FOLDS = args.folds
BATCH_SIZE = args.batch_size
DROP_OUT = args.drop_out
ALPHA = args.alpha
LAYERS = args.layers
HIDDEN = args.hidden
OUT_FEATURES = args.out_features
N_CLASSES = args.n_classes
SUPPRESS_IHC = args.suppress_ihc

# Print the arguments
print(f"Suppress Threshold: {args.suppress_threshold}")
print(f"Attention Heads: {args.attention_heads}")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Folds: {args.folds}")
print(f"Batch Size: {args.batch_size}")
print(f"Dropout: {args.drop_out}")
print(f"Alpha: {args.alpha}")
print(f"Layers: {args.layers}")
print(f"Hidden Features: {args.hidden}")
print(f"Output Features: {args.out_features}")
print(f"Number of Classes: {args.n_classes}")
print(f"Suppress non-homotopic interhemispherical edges: {args.suppress_ihc}")
print(f"Use asymmetry index for homotopic edges: {args.use_asym}")
print(f"Use curvature as additional features for edges: {args.use_curvature}")

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





ppmi_dataset = torch.load(DATASET_PATH)
connectivity_matrices = ppmi_dataset['data'].numpy()
connectivity_labels = ppmi_dataset['class_label'].numpy().reshape(-1, 1)
curvatures = ppmi_dataset['curvatures'].numpy()
asymmetry_indices = np.array([
        calculate_inter_hemispheric_asymmetry_vector_aal116(matrix) 
        for matrix in tqdm(connectivity_matrices, desc='Computing asymmetry indices')
    ])
if args.center_matrices:
    connectivity_matrices = center_matrices(connectivity_matrices)
    curvatures = center_matrices(curvatures)
    asymmetry_indices = center_matrices(asymmetry_indices)
    print(f'Centered Data')
elif args.normalize_matrices:
    connectivity_matrices = normalize_matrices(connectivity_matrices)
    curvatures = normalize_matrices(curvatures)
    asymmetry_indices = normalize_matrices(asymmetry_indices)
    print(f'Normalized Data')
elif args.standardize_matrices:
    connectivity_matrices = standardize_matrices(connectivity_matrices)
    curvatures = standardize_matrices(curvatures)
    asymmetry_indices = standardize_matrices(asymmetry_indices)
    print(f'Standardized Data')
print(f'Loaded {len(connectivity_matrices)} connectivity matrices and labels')
print(f'Loaded {len(curvatures)} curvatures')
print(f'Computed {len(asymmetry_indices)} asymmetry indices')



def suppress_interhemispheric_connectivity(matrix):
    """
    Suppresses the interhemispheric connections of the given connectivity matrix.
    """
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

if SUPPRESS_IHC:
    connectivity_matrices = np.array([suppress_interhemispheric_connectivity(matrix) for matrix in connectivity_matrices])
    print(f'{len(connectivity_matrices)} matrices got non-homotopic interhemispherical connections suppression')

def homotopic(i, j):
    return i//2 == j//2

def construct_graph(matrix, curvatures=[], asymmetry_indices=[], threshold=0.3):
    G = nx.Graph()
    num_nodes = matrix.shape[0]
    # Add nodes with connectivity strength and hemisphere as features
    for i in range(num_nodes):
        hemisphere = 0 if i % 2 == 0 else 1  # Assuming alternating arrangement
        G.add_node(i, strength=matrix[i, :].mean(), hemisphere=hemisphere)
    
    # Add edges with connectivity attribute and inter/intra-hemispheric features
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if abs(matrix[i, j]) > threshold:
                interhemispheric = 1 if (i % 2 != j % 2) else 0
                curvature = curvatures[i][j] if args.use_curvature else 0.0
                G.add_edge(i, j, weight=matrix[i, j], interhemispheric=interhemispheric, curvature=curvature, asymmetry_index=asymmetry_indices[i//2] if homotopic(i,j) and args.use_asym else 0.0)
    # Add additional node features
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree[node]
    clustering_coeffs = nx.clustering(G)
    for node, coeff in clustering_coeffs.items():
        G.nodes[node]['clustering'] = coeff
    betweenness = nx.betweenness_centrality(G)
    for node, centrality in betweenness.items():
        G.nodes[node]['betweenness'] = centrality
    # Add additional edge features
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
    # Extract node features
    node_features = np.array([[G.nodes[i]['strength'], G.nodes[i]['hemisphere'], G.nodes[i]['degree'], G.nodes[i]['clustering'], G.nodes[i]['betweenness']] for i in G.nodes])
    x = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

graphs = []
for i in range(len(connectivity_matrices)):
    graphs.append(construct_graph(connectivity_matrices[i], curvatures[i], asymmetry_indices[i//2], SUPPRESS_THRESHOLD))

connectome = graphs[STUDY_CONNECTOME_INDEX]

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

graph_data_list = [convert_to_pyg_data(G) for G in graphs]
labels = torch.tensor(connectivity_labels, dtype=torch.long)
data_loaders = create_data_loaders(graph_data_list, labels, n_splits=FOLDS, batch_size=BATCH_SIZE)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in tqdm(train_loader, desc='Training', leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))  # Flatten the target labels
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', leave=False):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1))  # Flatten the target labels
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y.view(-1)).sum().item()
    return total_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

# Training with Cross-Validation
for fold, (train_loader, test_loader) in enumerate(data_loaders):
    model = GATWithEdgeFeatures(
        in_features=NODE_FEATURES, 
        hidden_features=HIDDEN, 
        out_features=OUT_FEATURES, 
        edge_features=EDGE_FEATURES, 
        heads=ATTENTION_HEADS, 
        dropout=DROP_OUT, 
        alpha=ALPHA, 
        num_classes=N_CLASSES, 
        num_layers=LAYERS
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_model_wts = None

    for epoch in range(EPOCHS):  # Adjust the number of epochs as needed
        start_time = time.time()
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        elapsed_time = time.time() - start_time
        print(f'[Fold {fold + 1}][Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {elapsed_time:.2f}s')
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()
    
    # Save the best model for this fold
    torch.save({
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, f'models/GAT{"_suppressed" if SUPPRESS_IHC else ""}_{LAYERS}_{HIDDEN}_{ATTENTION_HEADS}_{fold}.pth')
