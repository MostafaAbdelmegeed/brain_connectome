import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import argparse
import random
import math

def parse_args():
    parser = argparse.ArgumentParser(description='Co-embed Data Preparation')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name')
    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    parser.add_argument('--span', type=float, default=0.04, help='Span for randomizing the threshold')
    parser.add_argument('--n_augments', type=int, default=1, help='Number of augmented samples to generate for the minority class')
    parser.add_argument('--gpu_id', type=int, default=0, help='Device to use')
    return parser.parse_args()

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
    n_adj = node_adjacency_matrix(new_connectivity).to(device)
    e_adj = edge_adjacency_matrix(n_adj, device)
    t = transition_matrix(n_adj, device)
    return new_connectivity, n_adj, e_adj, t

def augment_minority_class(connectivity, original_connectivity, num_augments=1, device=None, percentile=0.9, span=0.04):
    """Generate augmented samples for the minority class by varying the percentile thresholds and adding/removing edges."""
    augmented_samples = []
    
    abs_conn = torch.abs(connectivity)
    abs_original_conn = torch.abs(original_connectivity)  # Absolute values of the original matrix

    for _ in range(num_augments):
        randomized_span = random.uniform(-span/2, span/2)
        # Generate the threshold based on the original connectivity matrix
        lower_threshold = torch.quantile(abs_original_conn, percentile+randomized_span)
        mid_threshold = torch.quantile(abs_original_conn, percentile+randomized_span)
        upper_threshold = torch.quantile(abs_original_conn, 1.0)


        # Suppress edges below the lower threshold in the current connectivity matrix
        suppressed_conn = torch.where(abs_conn > mid_threshold, connectivity, torch.zeros_like(connectivity))
        
        # Add edges from the original matrix that were removed but are within the percentile range
        added_edges_conn = torch.where(
            (abs_original_conn <= upper_threshold) & (abs_original_conn > lower_threshold),
            original_connectivity,
            suppressed_conn
        )
        
        # Randomly decide to add or remove edges in the given percentile range
        add_or_remove = random.choice(['add', 'remove'])
        if add_or_remove == 'add':
            new_connectivity = added_edges_conn
        else:
            new_connectivity = suppressed_conn
        
        augmented_samples.append(new_connectivity.to(device))

    return augmented_samples

def process_and_augment(dataloader, device, percentile=0.9, num_augments=1, span=0.04):
    print("Processing and augmenting data...")

    # Calculate current class distribution
    label_counts = {}
    for _, label in dataloader:
        label = label.item()
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    label_counts = dict(sorted(label_counts.items()))

    print(f'Labels Count: {label_counts}')

    
    
    # Determine the target number of samples (equal to the maximum class count)
    target_count = max(label_counts.values())
    print(f'Target Count: {target_count}')

    n_augments = [round((target_count - n)/n) for _, n in label_counts.items()]
    print(f'Number of Augments: {n_augments}')

    new_connectivity_list = []
    node_adj_list = []
    edge_adj_list = []
    trans_list = []
    label_list = []

    for connectivity, label in tqdm(dataloader, desc="Processing matrices"):
        connectivity = connectivity[0].squeeze(0).to(device)
        # Generate original and augmented samples
        new_conn, n_adj, e_adj, t = coembed_pipeline(connectivity, device=device, percentile=percentile)
        new_connectivity_list.append(new_conn.cpu())
        node_adj_list.append(n_adj.cpu())
        edge_adj_list.append(e_adj.cpu())
        trans_list.append(t.cpu())
        label_list.append(label.cpu())
        
        # Augment if this is a minority class sample and it needs more samples
        if label_counts[label.item()] < target_count:
            augmented_connectivities = augment_minority_class(new_conn, connectivity, num_augments=n_augments[label.item()], device=device, percentile=percentile, span=span)
            for aug_conn in augmented_connectivities:
                aug_n_adj = node_adjacency_matrix(aug_conn).to(device)
                aug_e_adj = edge_adjacency_matrix(aug_n_adj, device)
                aug_t = transition_matrix(aug_n_adj, device)
                
                new_connectivity_list.append(aug_conn.cpu())
                node_adj_list.append(aug_n_adj.cpu())
                edge_adj_list.append(aug_e_adj.cpu())
                trans_list.append(aug_t.cpu())
                label_list.append(label.cpu())
            # Update the count of the augmented class
            label_counts[label.item()] += len(augmented_connectivities)

    return new_connectivity_list, node_adj_list, edge_adj_list, trans_list, label_list



def main():
    args = parse_args()
    dataset_name = args.dataset
    dataset_path = args.dataset_path
    if not dataset_path:
        data = torch.load(f'data/{dataset_name}.pth')
    else:
        data = torch.load(dataset_path)
        dataset_name = 'ppmi' if 'ppmi' in dataset_path else 'adni'
    span = args.span
    connectivity = data['matrix']
    label = data['label']
    n_augments = args.n_augments
    # Ensure connectivity is 3D: (num_samples, num_nodes, num_nodes)
    if connectivity.dim() == 2:
        connectivity = connectivity.unsqueeze(0)
    elif connectivity.dim() != 3:
        raise ValueError("Connectivity matrix must be 2D or 3D")

    print(f'Labels Unique: {label.unique()}')

    # Prepare DataLoader
    tensor_dataset = TensorDataset(connectivity, label)
    dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    new_connectivity, node_adj, edge_adj, trans, labels = process_and_augment(dataloader, device, args.percentile, num_augments=n_augments, span=span)
    # new_connectivity = [x for xs in new_connectivity for x in xs]
    # node_adj = [x for xs in node_adj for x in xs]
    # edge_adj = [x for xs in edge_adj for x in xs]
    # trans = [x for xs in trans for x in xs]
    # labels = [x for xs in labels for x in xs]
    # new_connectivity = torch.stack(new_connectivity)
    # node_adj = torch.stack(node_adj)
    # edge_adj = torch.stack(edge_adj)
    # trans = torch.stack(trans)
    labels = torch.stack(labels).squeeze(1)


    print(f"Number of samples per class: {np.bincount(np.array(labels))}")
    print(f"Total number of labels: {labels.shape}")
    print(f'Unique Labels: {labels.unique()}')
    print(f'Number of Connectivity matrices: {len(new_connectivity)}')
    print(f'Number of Node adjacency matrices: {len(node_adj)}')
    print(f'Number of Edge adjacency matrices: {len(edge_adj)}')
    print(f'Number of Transition matrices: {len(trans)}')
    print("Saving data...")
    if not dataset_name:
        dataset_name = 'ppmi' if 'ppmi' in dataset_path else 'adni'
    path = f'data/{dataset_name}_coembed_p{int(args.percentile*100)}_augmented.pth'
    torch.save({'connectivity': new_connectivity, 'node_adj': node_adj, 
                'edge_adj': edge_adj, 'transition': trans, 'label': labels}, 
                path)
    
    print(f"Data saved to data/{dataset_name}_coembed_p{int(args.percentile*100)}_augmented.pth")

if __name__ == '__main__':
    main()
