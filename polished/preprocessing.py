import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import argparse
import random
import sys
from torch.utils.data import Sampler

is_interactive = sys.stdout.isatty()


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    values = sparse_mx.data.astype(np.float32)
    shape = torch.Size(sparse_mx.shape)
    indices_tensor = torch.tensor(indices, dtype=torch.int64, device=device)
    values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
    sparse_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, shape)
    return sparse_tensor.coalesce()  # Ensure the tensor is coalesced

def node_adjacency_matrix(connectivity):
    """Create a node adjacency matrix from connectivity matrix."""
    return (connectivity != 0).float()


def edge_adjacency_matrix(node_adj, device):
    """Create an edge adjacency matrix from a node adjacency matrix, including self-loops."""
    node_adj = node_adj.cpu().numpy()

    # Find all edges, including self-loops
    edge_index = np.array(np.nonzero(node_adj))
    num_edge = edge_index.shape[1]

    # Initialize the edge adjacency matrix
    edge_adj = np.zeros((num_edge, num_edge))

    # Create edge adjacency matrix using broadcasting and vectorized operations
    for i in range(num_edge):
        # Check if the edges share a common node or are the same edge (self-loop)
        common_nodes = (edge_index[:, i][:, None] == edge_index).any(axis=0)
        edge_adj[i, common_nodes] = 1

    # The diagonal is set to 1, explicitly indicating self-loops
    np.fill_diagonal(edge_adj, 1)

    return sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(edge_adj), device=device)


def transition_matrix(node_adj, device):
    """Create a transition matrix from a node adjacency matrix, including self-loops."""
    node_adj = node_adj.cpu().numpy()

    # Find all edges, including self-loops
    edge_index = np.array(np.nonzero(node_adj))
    num_edge = edge_index.shape[1]
    
    # Each edge connects two nodes, hence repeat each edge index twice
    col_index = np.repeat(np.arange(num_edge), 2)
    
    # The row index corresponds to the node indices connected by each edge
    row_index = np.hstack([edge_index[0], edge_index[1]])
    
    # Data array indicates the presence of a connection
    data = np.ones(num_edge * 2)

    # The transition matrix has shape (number of nodes, number of edges)
    T = sp.csr_matrix((data, (row_index, col_index)), shape=(node_adj.shape[0], num_edge))
    
    return sparse_mx_to_torch_sparse_tensor(T, device=device)

def suppress_below_percentile(x, percentile):
    abs_x = torch.abs(x)
    threshold = torch.quantile(abs_x, percentile)
    return torch.where(abs_x > threshold, x, torch.zeros_like(x))

def coembed_pipeline(connectivity, percentile=0.9, device=None):
    new_connectivity = suppress_below_percentile(connectivity, percentile).to(device)
    # n_adj = node_adjacency_matrix(new_connectivity).to(device)
    # e_adj = edge_adjacency_matrix(n_adj, device)
    # t = transition_matrix(n_adj, device)
    return new_connectivity#, n_adj, e_adj, t

def augment_minority_class(connectivity, original_connectivity, num_augments=1, device=None, percentile=0.9, span=0.04, max_mods=50):
    """Generate augmented samples for the minority class by randomly adding/removing edges."""
    augmented_samples = []
    # edge_counts = []
    
    abs_conn = torch.abs(connectivity)
    abs_original_conn = torch.abs(original_connectivity)  # Absolute values of the original matrix
    # print(f'--------- Number of edges in origin: {torch.sum(abs_original_conn > 0)}')
    # print(f'--------- Number of edges in source: {torch.sum(abs_conn > 0)}')
    for _ in range(num_augments):
        randomized_span = random.uniform(-span, 0)
        # Generate the threshold based on the original connectivity matrix
        lower_threshold = torch.quantile(abs_original_conn, percentile+randomized_span*2)
        # print(f'Lower percentile: {percentile+randomized_span}, lower_threshold: {lower_threshold}')
        upper_threshold = torch.quantile(abs_original_conn, 1.0+randomized_span)
        # print(f'Upper percentile: {1.0+randomized_span}, upper_threshold: {upper_threshold}')
        print(f'Min value: {torch.min(abs_original_conn)}, Lower threshold: {lower_threshold}, Upper threshold: {upper_threshold}, Max value: {torch.max(abs_original_conn)}')

        # Generate all possible (i, j) pairs excluding self-loops (i == j)
        indices = [(i, j) for i in range(connectivity.size(0)) for j in range(i+1, connectivity.size(1))]
        # Start with a copy of the original connectivity matrix
        augmented_conn = connectivity.clone()
        add_mods = 0
        remove_mods = 0
        # Iterate over all elements in the connectivity matrix
        # Shuffle indices for random order
        random.shuffle(indices)
        for i, j in indices:
            if i != j:  # Avoid self-loops if your context requires it
                edge_strength = abs_conn[i, j]
                original_edge_strength = abs_original_conn[i, j]
                # Decide randomly to add or remove the edge
                if random.random() < 0.9:  # Randomly decide to add or remove
                    # Add the edge if it was suppressed but within the percentile range
                    if edge_strength < lower_threshold and lower_threshold <= original_edge_strength <= upper_threshold:
                        augmented_conn[i, j] = original_connectivity[i, j]
                        augmented_conn[j, i] = original_connectivity[j, i]
                        add_mods += 1
                else:
                    # Remove the edge if it's above the threshold
                    if edge_strength > lower_threshold and lower_threshold <= original_edge_strength <= upper_threshold:
                        augmented_conn[i, j] = 0
                        augmented_conn[j, i] = 0
                        remove_mods += 1
            if add_mods+remove_mods >= max_mods:
                break
        # num_edges = torch.sum(torch.abs(augmented_conn) > 0).item()
        # edge_counts.append(num_edges)
        # print(f'Number of edges in augmented: {num_edges}')
        # print(f'Number of added edges: {add_mods}, number of removed edges: {remove_mods}')
        augmented_samples.append(augmented_conn.to(device))
    
    # Calculate the mean number of edges per augmented sample
    # mean_num_edges = sum(edge_counts) / len(edge_counts) if edge_counts else 0
    # print(f'============= Mean number of edges per augmented sample: {mean_num_edges}')
    return augmented_samples


def process(dataloader, device, percentile=0.9, span=0.04, augment=False):
    label_counts = {}
    # Calculate current class distribution
    for _, label in dataloader:
        label = label.item()
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
    label_counts = dict(sorted(label_counts.items()))
    # Determine the target number of samples (equal to the maximum class count)
    target_count = max(label_counts.values())
    n_augments = [round((target_count - n)/n) for _, n in label_counts.items()]


    new_connectivity_list = []
    # node_adj_list = []
    # edge_adj_list = []
    # trans_list = []
    label_list = []

    for connectivity, label in tqdm(dataloader, desc="Processing matrices", disable=not is_interactive):
        connectivity = connectivity[0].squeeze(0).to(device)
        # Generate original and augmented samples
        # new_conn, n_adj, e_adj, t = coembed_pipeline(connectivity, device=device, percentile=percentile)
        new_conn = coembed_pipeline(connectivity, device=device, percentile=percentile)
        new_connectivity_list.append(new_conn)
        # node_adj_list.append(n_adj.cpu())
        # edge_adj_list.append(e_adj.cpu())
        # trans_list.append(t.cpu())
        label_list.append(label)
        

        # Augment if this is a minority class sample and it needs more samples
        if augment and (label_counts[label.item()] < target_count):
            augmented_connectivities = augment_minority_class(new_conn, connectivity, num_augments=n_augments[label.item()], device=device, percentile=percentile, span=span)
            for aug_conn in augmented_connectivities:
                # aug_n_adj = node_adjacency_matrix(aug_conn).to(device)
                # aug_e_adj = edge_adjacency_matrix(aug_n_adj, device)
                # aug_t = transition_matrix(aug_n_adj, device)
                
                new_connectivity_list.append(aug_conn.cpu())
                # node_adj_list.append(aug_n_adj.cpu())
                # edge_adj_list.append(aug_e_adj.cpu())
                # trans_list.append(aug_t.cpu())
                label_list.append(label.cpu())
            # Update the count of the augmented class
            label_counts[label.item()] += len(augmented_connectivities)
            # Convert augmented lists to tensors directly

    return {
        'connectivity': new_connectivity_list,
        # 'node_adj': node_adj_list,
        # 'edge_adj': edge_adj_list,
        # 'transition': trans_list,
        'label': label_list
    }




class AugmentingSampler(Sampler):
    def __init__(self, data, labels, num_samples=None, device='cpu', percentile=0.9, span=0.04, max_mods=50):
        self.labels = labels
        self.data = data
        self.device = device
        self.percentile = percentile
        self.span = span
        self.max_mods = max_mods
        # Determine the number of samples each class should have
        self.label_counts = np.bincount(self.labels.numpy())
        self.target_count = max(self.label_counts)
        self.num_samples = num_samples or sum(self.label_counts)

    def __iter__(self):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        # For each class, calculate how many times we need to sample each data point
        weights = [self.target_count // self.label_counts[label] for label in self.labels]
        # Weighted sampling without replacement
        sampled_indices = np.random.choice(indices, size=self.num_samples, replace=True, p=weights/np.sum(weights))
        # Augment data dynamically
        for idx in sampled_indices:
            data_point = self.data[idx]
            label = self.labels[idx]
            if self.label_counts[label] < self.target_count:
                # Perform augmentation
                data_point = self.augment_data(data_point)
            yield data_point.to(self.device)

    def augment_data(self, connectivity):
        # Perform augmentation logic (simplified)
        randomized_span = random.uniform(-self.span, 0)
        lower_threshold = torch.quantile(torch.abs(connectivity), self.percentile + randomized_span * 2)
        upper_threshold = torch.quantile(torch.abs(connectivity), 1.0 + randomized_span)
        indices = [(i, j) for i in range(connectivity.size(0)) for j in range(i + 1, connectivity.size(1))]
        random.shuffle(indices)
        add_mods = 0
        remove_mods = 0
        for i, j in indices:
            if add_mods + remove_mods >= self.max_mods:
                break
            if random.random() < 0.5:  # Random choice to add or remove connections
                connectivity[i, j] = connectivity[j, i] = random.uniform(lower_threshold, upper_threshold)
                add_mods += 1
            else:
                connectivity[i, j] = connectivity[j, i] = 0
                remove_mods += 1
        return connectivity

    def __len__(self):
        return self.num_samples
