import torch
from torch_geometric.data import Data, Dataset, Batch


class VanillaDataset(Dataset):
    def __init__(self, data):
        self.connectivities = data['connectivity']
        self.node_adjacencies = data['node_adj']
        self.edge_adjacencies = data['edge_adj']
        self.transition = data['transition']
        self.labels = data['label']
        self.node_num = self.connectivities[0].shape[0]

    def node_count(self):
        return self.node_num

    def edge_count(self):
        return self.edge_adjacencies[0].shape[0]
    
    def unique_labels(self):
        return torch.unique(self.labels)
    
    def num_classes(self):
        return len(self.unique_labels())
    
    def edge_attr_dim(self):
        return 1

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        node_adj = self.node_adjacencies[idx]
        edge_adj = self.edge_adjacencies[idx]
        transition = self.transition[idx]

        # Ensure coalesced
        edge_adj = edge_adj.coalesce()
        transition = transition.coalesce()
        label = self.labels[idx]

        # Initialize lists for edge_index and edge_attr
        edge_index = []
        edge_attr = []

        # Populate edge_index and edge_attr with additional features
        for j in range(connectivity.shape[0]):
            for k in range(connectivity.shape[0]):
                if node_adj[j, k] == 0:
                    continue
                edge_index.append([j, k])
                # Include additional edge attributes like isInter, isLeftIntra, etc.
                edge_attr.append([
                    connectivity[j, k], 
                ])
        
        # Convert lists to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(
            x=torch.eye(connectivity.shape[0], dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label.type(torch.long),
            num_nodes=connectivity.shape[0],
            node_adj=node_adj,
            edge_adj=edge_adj,
            transition=transition
        )
        
        return data


class BrainDataset(Dataset):
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
    
    def node_count(self):
        return self.node_num

    def edge_count(self):
        return self.edge_adjacencies[0].shape[0]
    
    def unique_labels(self):
        return torch.unique(self.labels)
    
    def num_classes(self):
        return len(self.unique_labels())
    
    def edge_attr_dim(self):
        return 5

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        node_adj = self.node_adjacencies[idx]
        edge_adj = self.edge_adjacencies[idx]
        transition = self.transition[idx]

        # Ensure coalesced
        edge_adj = edge_adj.coalesce()
        transition = transition.coalesce()
        label = self.labels[idx]

        # Initialize lists for edge_index and edge_attr
        edge_index = []
        edge_attr = []

        # Populate edge_index and edge_attr with additional features
        for j in range(connectivity.shape[0]):
            for k in range(connectivity.shape[0]):
                if node_adj[j, k] == 0:
                    continue
                edge_index.append([j, k])
                # Include additional edge attributes like isInter, isLeftIntra, etc.
                edge_attr.append([
                    connectivity[j, k], 
                    self.isInter(j, k), 
                    self.isLeftIntra(j, k), 
                    self.isRightIntra(j, k), 
                    self.isHomo(j, k)
                ])
        
        # Convert lists to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(
            x=torch.eye(connectivity.shape[0], dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label.type(torch.long),
            num_nodes=connectivity.shape[0],
            node_adj=node_adj,
            edge_adj=edge_adj,
            transition=transition
        )
        
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

def collate_function(batch):
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
        edge_adj = pad_matrix(data.edge_adj.to_dense(), max_edges)
        transition = pad_matrix(data.transition.to_dense(), max_nodes)  
        batched_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, num_nodes=max_nodes,
                                 node_adj=node_adj, edge_adj=edge_adj, transition=transition.to_sparse()))
    return Batch.from_data_list(batched_data)


