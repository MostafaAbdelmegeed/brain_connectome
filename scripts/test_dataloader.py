import torch
from torch_geometric.data import Data, Batch, Dataset
from torch_geometric.loader import DataLoader

class ConnectivityDataset(Dataset):
    def __init__(self, data):
        self.connectivities = data['connectivity']
        self.node_adjacencies = data['node_adj']
        self.edge_adjacencies = data['edge_adj']
        self.transition = data['transition']
        self.labels = data['label']
        self.node_num = self.connectivities[0].shape[0]
        self.left_indices, self.right_indices = self.get_hemisphere_indices()
    
    def get_hemisphere_indices(self):
        left_indices = [i for i in range(self.node_num) if i % 2 == 0]
        right_indices = [i for i in range(self.node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self, i, j):
        return (i in self.left_indices and j in self.right_indices) or (i in self.right_indices and j in self.left_indices)

    def isLeftIntra(self, i, j):
        return i in self.left_indices and j in self.left_indices

    def isRightIntra(self, i, j):
        return i in self.right_indices and j in self.right_indices

    def isHomo(self, i, j):
        return i // 2 == j // 2 and abs(i - j) == 1
    
    def edge_features_count(self):
        return 5
    
    def node_features_count(self):
        return 116

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        print(f'Fetching item {idx}')  # Debug print statement
        connectivity = self.connectivities[idx]
        node_adj = self.node_adjacencies[idx]
        edge_adj = self.edge_adjacencies[idx]
        transition = self.transition[idx]
        edge_adj = edge_adj.coalesce()
        transition = transition.coalesce()
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        for j in range(connectivity.shape[0]):
            for k in range(j + 1, connectivity.shape[0]):
                if node_adj[j, k] == 0:
                    continue
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isLeftIntra(j, k), self.isRightIntra(j, k), self.isHomo(j, k)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = label.type(torch.long)
        data = Data(x=torch.eye(connectivity.shape[0], dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=connectivity.shape[0], node_adj=node_adj, edge_adj=edge_adj, transition=transition)
        return data

def pad_edge_index(edge_index, max_edges):
    pad_size = max_edges - edge_index.size(1)
    if pad_size > 0:
        pad = torch.zeros((2, pad_size), dtype=edge_index.dtype, device=edge_index.device)
        edge_index = torch.cat([edge_index, pad], dim=1)
    return edge_index

def pad_edge_attr(edge_attr, max_edges, feature_dim):
    pad_size = max_edges - edge_attr.size(0)
    if pad_size > 0:
        pad = torch.zeros((pad_size, feature_dim), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, pad], dim=0)
    return edge_attr

def pad_matrix(matrix, max_nodes):
    pad_size = max_nodes - matrix.size(0)
    if pad_size > 0:
        pad = torch.zeros((pad_size, matrix.size(1)), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad], dim=0)
        pad = torch.zeros((matrix.size(0), pad_size), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad], dim=1)
    return matrix

def custom_collate(batch):
    print(f'collate_function is called with batch size: {len(batch)}')  # Debug print statement
    max_nodes = max([data.num_nodes for data in batch])
    max_edges = max([data.edge_index.size(1) for data in batch])
    feature_dim = batch[0].edge_attr.size(1)
    batched_data = []
    for data in batch:
        x = torch.eye(max_nodes, dtype=data.x.dtype, device=data.x.device)
        x[:data.num_nodes, :data.num_nodes] = data.x
        
        edge_index = pad_edge_index(data.edge_index, max_edges)
        edge_attr = pad_edge_attr(data.edge_attr, max_edges, feature_dim)
        
        node_adj = pad_matrix(data.node_adj, max_nodes)
        edge_adj = pad_matrix(data.edge_adj.to_dense(), max_nodes)
        transition = pad_matrix(data.transition.to_dense(), max_nodes)  
        batched_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, num_nodes=max_nodes,
                                 node_adj=node_adj, edge_adj=edge_adj.to_sparse(), transition=transition.to_sparse()))
    return Batch.from_data_list(batched_data)

dataset_name = 'adni'
# Ensure the file path is correct
file_path = f'data/{dataset_name}_coembed_p{int(0.95*100)}.pth'
print(f'Loading data from: {file_path}')
data = torch.load(file_path)

# Check if data is loaded correctly
print(f'Data keys: {data.keys()}')
print(f'Number of samples in dataset: {len(data["connectivity"])}')

dataset = ConnectivityDataset(data)
print(f'Dataset length: {len(dataset)}')  # Debug print statement

train_loader = DataLoader(dataset, collate_fn=custom_collate, batch_size=2, shuffle=True)
# print(f'Length of DataLoader: {len(train_loader)}')  # Debug print statement
# print(f'Batch size of DataLoader: {train_loader.batch_size}')  # Debug print statement
# print(f'Shuffle of DataLoader: {train_loader.shuffle}')  # Debug print statement
# print(f'Number of workers of DataLoader: {train_loader.num_workers}')  # Debug print statement
# print(f'Pin memory of DataLoader: {train_loader.pin_memory}')  # Debug print statement
# print(f'Drop last of DataLoader: {train_loader.drop_last}')  # Debug print statement
# print(f'Timeout of DataLoader: {train_loader.timeout}')  # Debug print statement
# print(f'Worker init fn of DataLoader: {train_loader.worker_init_fn}')  # Debug print statement
# print(f'Multiprocessing context of DataLoader: {train_loader.multiprocessing_context}')  # Debug print statement
# print(f'Generator of DataLoader: {train_loader.generator}')  # Debug print statement
# print(f'Prefetch factor of DataLoader: {train_loader.prefetch_factor}')  # Debug print statement
# print(f'Persistent_workers of DataLoader: {train_loader.persistent_workers}')  # Debug print statement
# print(f'Pinned memory of DataLoader: {train_loader.pinned_memory}')  # Debug print statement
# print(f'Sampler of DataLoader: {train_loader.sampler}')  # Debug print statement
# print(f'Batch sampler of DataLoader: {train_loader.batch_sampler}')  # Debug print statement
train_loader.collate_fn = custom_collate
# Iterate through the DataLoader to verify collate function is called
for batch_idx, data in enumerate(train_loader):
    print(f'Batch {batch_idx}: {data}')