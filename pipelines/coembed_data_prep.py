import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Co-embed Data Preparation')
    parser.add_argument('--dataset', type=str, default='ppmi', help='Dataset name')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
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
    """Create an edge adjacency matrix from a node adjacency matrix."""
    node_adj = node_adj.cpu().numpy()
    np.fill_diagonal(node_adj, 0)
    edge_index = np.triu(node_adj).nonzero()
    num_edge = len(edge_index[0])
    edge_adj = np.zeros((num_edge, num_edge))

    # Create edge adjacency matrix
    for i in range(num_edge):
        for j in range(i + 1, num_edge):
            # Check if the edges share a common node
            if (edge_index[0][i] == edge_index[0][j]) or (edge_index[1][i] == edge_index[1][j]) or (edge_index[0][i] == edge_index[1][j]) or (edge_index[1][i] == edge_index[0][j]):
                edge_adj[i, j] = 1
                edge_adj[j, i] = 1  # Because the adjacency matrix is symmetric

    # The diagonal should always be 1, indicating self-loops
    np.fill_diagonal(edge_adj, 1)

    return sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(edge_adj), device=device)

def transition_matrix(node_adj, device):
    """Create a transition matrix from a node adjacency matrix."""
    node_adj = node_adj.cpu().numpy()
    np.fill_diagonal(node_adj, 0)
    edge_index = np.triu(node_adj).nonzero()
    num_edge = len(edge_index[0])
    col_index = np.repeat(np.arange(num_edge), 2)
    row_index = np.hstack([edge_index[0], edge_index[1]])
    data = np.ones(num_edge * 2)
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

def process_matrices(dataloader, device, percentile=0.9):
    new_connectivity_list = []
    node_adj_list = []
    edge_adj_list = []
    trans_list = []

    for i, (connectivity, _) in enumerate(tqdm(dataloader, desc="Processing matrices")):
        connectivity = connectivity.squeeze(0).to(device)
        new_conn, n_adj, e_adj, t = coembed_pipeline(connectivity, device=device, percentile=percentile)
        new_connectivity_list.append(new_conn.cpu())
        node_adj_list.append(n_adj.cpu())
        edge_adj_list.append(e_adj.cpu())
        trans_list.append(t.cpu())

        # # Print message after each iteration to debug shapes
        # print(f"Processed {i + 1}/{len(dataloader)}")
        # print(f"new_conn shape: {new_conn.shape}, non-zero elements: {new_conn.nonzero().size(0)}")
        # print(f"n_adj shape: {n_adj.shape}, non-zero elements: {n_adj.nonzero().size(0)}")
        # print(f"e_adj shape: {e_adj.shape}, non-zero elements: {e_adj._nnz()}")
        # print(f"t shape: {t.shape}, non-zero elements: {t._nnz()}")

    new_connectivity = torch.stack(new_connectivity_list)
    node_adj = torch.stack(node_adj_list)
    edge_adj = torch.stack(edge_adj_list)
    trans = torch.stack(trans_list)

    return new_connectivity, node_adj, edge_adj, trans

def main():
    args = parse_args()
    dataset_name = args.dataset
    data = torch.load(f'data/{dataset_name}.pth')
    connectivity = data['matrix']
    label = data['label'].to('cuda')

    # Ensure connectivity is 3D: (num_samples, num_nodes, num_nodes)
    if connectivity.dim() == 2:
        connectivity = connectivity.unsqueeze(0)
    elif connectivity.dim() != 3:
        raise ValueError("Connectivity matrix must be 2D or 3D")

    # Prepare DataLoader
    tensor_dataset = TensorDataset(connectivity, label)
    dataloader = DataLoader(tensor_dataset, batch_size=1, shuffle=False, num_workers=4)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    new_connectivity, node_adj, edge_adj, trans = process_matrices(dataloader, device, args.percentile)

    label = label.float()
    torch.save({'connectivity': new_connectivity, 'node_adj': node_adj, 
                'edge_adj': edge_adj, 'transition': trans, 'label': label}, 
                f'data/{dataset_name}_coembed_p{int(args.percentile*100)}.pth')
    
    print("Data saved successfully!")

if __name__ == '__main__':
    main()
