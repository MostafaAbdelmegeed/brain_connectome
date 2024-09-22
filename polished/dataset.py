import torch
from torch_geometric.data import Data, Dataset, Batch
import os
from scipy.io import loadmat
from torch.utils.data import Dataset as TorchDataset
from help_funs import *
import networkx as nx


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


class PPMIBrainDataset(Dataset):
    def __init__(self, root_dir):
        super(PPMIBrainDataset, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    data = loadmat(file_path)
                    features = data['data']
                    sentence_sizes.append(features.shape[0]) 
                    label = self.get_label(subdir)
                    self.data.append(features)
                    self.labels.append(label)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 2
        elif 'swedd' in subdir:
            return 3
        else:
            print("Label error")
            return -1  
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.corrcoef(torch.tensor(data).squeeze().T)
        data = torch.nan_to_num(data)
        edge_index, edge_attr = from_scipy_sparse_matrix(sp.coo_matrix(data))
        data = Data(x=data.float(), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        return data
# class BrainDataset(Dataset):
#     def __init__(self, data):
#         self.connectivities = data['connectivity']
#         self.node_adjacencies = data['node_adj']
#         self.edge_adjacencies = data['edge_adj']
#         self.transition = data['transition']
#         self.labels = data['label']
#         self.node_num = self.connectivities[0].shape[0]
#         self.left_indices, self.right_indices = self.get_hemisphere_indices()
    
#     def get_hemisphere_indices(self):
#         left_indices = [i for i in range(self.node_num) if i % 2 == 0]
#         right_indices = [i for i in range(self.node_num) if i % 2 != 0]
#         return left_indices, right_indices

#     def isInter(self, i, j):
#         return (i in self.left_indices and j in self.right_indices) or (i in self.right_indices and j in self.left_indices)

#     def isLeftIntra(self, i, j):
#         return i in self.left_indices and j in self.left_indices

#     def isRightIntra(self, i, j):
#         return i in self.right_indices and j in self.right_indices

#     def isHomo(self, i, j):
#         return i // 2 == j // 2 and abs(i - j) == 1
    
#     def node_count(self):
#         return self.node_num

#     def edge_count(self):
#         return self.edge_adjacencies[0].shape[0]
    
#     def unique_labels(self):
#         return torch.unique(self.labels)
    
#     def num_classes(self):
#         return len(self.unique_labels())
    
#     def edge_attr_dim(self):
#         return 5

#     def __len__(self):
#         return len(self.connectivities)

#     def __getitem__(self, idx):
#         connectivity = self.connectivities[idx]
#         node_adj = self.node_adjacencies[idx]
#         edge_adj = self.edge_adjacencies[idx]
#         transition = self.transition[idx]
#         label = self.labels[idx]

#         # Ensure coalesced
#         edge_adj = edge_adj.coalesce()
#         transition = transition.coalesce()
#         num_nodes = connectivity.shape[0]

#         # Initialize lists for edge_index and edge_attr
#         edge_index = []
#         edge_attr = []

#         # Initialize node degree, strength, and hemisphere indicator tensors
#         node_degree = torch.zeros(num_nodes, dtype=torch.float)
#         node_strength = torch.zeros(num_nodes, dtype=torch.float)
#         hemisphere_indicator = torch.zeros(num_nodes, dtype=torch.float)
#         hemisphere_indicator[self.right_indices] = 1.0  # Right hemisphere nodes get 1.0

#         # Build adjacency list for clustering coefficient
#         adjacency_list = [[] for _ in range(num_nodes)]
#         # print(f'Connectivity min: {connectivity.min()}, max: {connectivity.max()}')
#         # Populate edge_index, edge_attr, and compute node features
#         for j in range(num_nodes):
#             for k in range(num_nodes):
#                 if node_adj[j, k] == 0:
#                     continue
#                 edge_index.append([j, k])
#                 edge_attr.append([
#                     connectivity[j, k],
#                     self.isInter(j, k),
#                     self.isLeftIntra(j, k),
#                     self.isRightIntra(j, k),
#                     self.isHomo(j, k)
#                 ])
#                 # Update node degree and strength
#                 node_degree[j] += 1
#                 node_strength[j] += connectivity[j, k]
#                 adjacency_list[j].append(k)

#         # Compute clustering coefficient for each node (optional)
#         node_clustering = torch.zeros(num_nodes, dtype=torch.float)
#         for i in range(num_nodes):
#             neighbors = adjacency_list[i]
#             if len(neighbors) < 2:
#                 node_clustering[i] = 0.0
#             else:
#                 links = 0
#                 for u in neighbors:
#                     for v in neighbors:
#                         if u != v and node_adj[u, v]:
#                             links += 1
#                 node_clustering[i] = links / (len(neighbors) * (len(neighbors) - 1))

        
#         # Separate continuous and binary features
#         node_features = torch.stack([
#             node_degree,
#             node_strength,
#             node_clustering
#         ], dim=1)

#         # Convert lists to tensors
#         edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
#         edge_attr = torch.tensor(edge_attr, dtype=torch.float)

#         data = Data(
#             x=node_features,
#             edge_index=edge_index,
#             edge_attr=edge_attr,
#             y=label.type(torch.long),
#             num_nodes=num_nodes,
#             node_adj=node_adj,
#             edge_adj=edge_adj,
#             transition=transition
#         )
#         return data


# def pad_edge_index(edge_index, max_edges):
#     pad_size = max_edges - edge_index.size(1)
#     if pad_size > 0:
#         pad = torch.zeros((2, pad_size), dtype=edge_index.dtype, device=edge_index.device)
#         edge_index = torch.cat([edge_index, pad], dim=1)
#     return edge_index

# def pad_edge_attr(edge_attr, max_edges, feature_dim):
#     pad_size = max_edges - edge_attr.size(0)
#     if pad_size > 0:
#         pad = torch.zeros((pad_size, feature_dim), dtype=edge_attr.dtype, device=edge_attr.device)
#         edge_attr = torch.cat([edge_attr, pad], dim=0)
#     return edge_attr

# def pad_matrix(matrix, max_nodes):
#     pad_size = max_nodes - matrix.size(0)
#     if pad_size > 0:
#         pad = torch.zeros((pad_size, matrix.size(1)), dtype=matrix.dtype, device=matrix.device)
#         matrix = torch.cat([matrix, pad], dim=0)
#         pad = torch.zeros((matrix.size(0), pad_size), dtype=matrix.dtype, device=matrix.device)
#         matrix = torch.cat([matrix, pad], dim=1)
#     return matrix

# def collate_function(batch):
#     max_nodes = max([data.num_nodes for data in batch])
#     max_edges = max([data.edge_index.size(1) for data in batch])
#     feature_dim = batch[0].edge_attr.size(1)
#     batched_data = []
#     for data in batch:
#         x = torch.eye(max_nodes, dtype=data.x.dtype, device=data.x.device)
#         x[:data.num_nodes, :data.num_nodes] = data.x
        
#         edge_index = pad_edge_index(data.edge_index, max_edges)
#         edge_attr = pad_edge_attr(data.edge_attr, max_edges, feature_dim)
        
#         node_adj = pad_matrix(data.node_adj, max_nodes)
#         edge_adj = pad_matrix(data.edge_adj.to_dense(), max_edges)
#         transition = pad_matrix(data.transition.to_dense(), max_nodes)  
#         batched_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, num_nodes=max_nodes,
#                                  node_adj=node_adj, edge_adj=edge_adj, transition=transition.to_sparse()))
#     return Batch.from_data_list(batched_data)


class Dataset_PPMI(TorchDataset):
    def __init__(self, root_dir):
        super(Dataset_PPMI, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'AAL116_features_timeseries' in file:
                    file_path = os.path.join(subdir, file)
                    data = loadmat(file_path)
                    features = data['data']
                    sentence_sizes.append(features.shape[0]) 
                    label = self.get_label(subdir)
                    self.data.append(features)
                    self.labels.append(label)

    def get_label(self, subdir):
        if 'control' in subdir:
            return 0
        elif 'patient' in subdir:
            return 1
        elif 'prodromal' in subdir:
            return 2
        elif 'swedd' in subdir:
            return 3
        else:
            print("Label error")
            return -1  

    def edge_attr_dim(self):
        return 1     

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.corrcoef(torch.tensor(data).squeeze().T)
        data = torch.nan_to_num(data)
        data = pearson_dataset(data, label, 10)
        return data
        # data = torch.tensor(self.data[idx]).float()
        # label = self.labels[idx]
        
        # # Compute the correlation matrix
        # corr_matrix = torch.corrcoef(data.squeeze().T)
        # corr_matrix = torch.nan_to_num(corr_matrix)
        
        # # Threshold the correlation matrix to create an adjacency matrix
        # adjacency_matrix = preprocess_adjacency_matrix(corr_matrix, 10)
        # # Since adjacency_matrix is a tuple, let's print its components
        # edge_index, edge_attr = from_scipy_sparse_matrix(adjacency_matrix)

        # # Now convert edge_index and edge_attr to graph
        # G = nx.Graph(adjacency_matrix)

        
        # # Compute eigenvector centrality
        # eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
         
        # # Create a feature vector where each node feature is its eigenvector centrality score
        # node_features = torch.tensor([eigenvector_centrality[i] for i in range(len(eigenvector_centrality))]).float().unsqueeze(1)  # Shape [num_nodes, 1]
        # # Add binary hemisphere feature: 0 for left hemisphere, 1 for right hemisphere
        # num_nodes = len(eigenvector_centrality)
        # hemisphere_feature = torch.zeros(num_nodes, 1)  # Initialize with 0 (left hemisphere)
        # hemisphere_feature[1::2] = 1  # Set to 1 for right hemisphere (alternating L, R)
        # # Concatenate eigenvector centrality and hemisphere features
        # node_features = torch.cat([node_features, hemisphere_feature], dim=1)  # Shape [num_nodes, 2]
        # # Create PyG data object with eigenvector centrality as node features
        # data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))



# class Dataset_PPMI(TorchDataset):
#     def __init__(self, root_dir):
#         super(Dataset_PPMI, self).__init__()
#         self.root_dir = root_dir
#         self.data = []
#         self.labels = []
#         self.load_data()

#     def load_data(self):
#         sentence_sizes = []
#         for subdir, _, files in os.walk(self.root_dir):
#             for file in files:
#                 if 'AAL116_features_timeseries' in file:
#                     file_path = os.path.join(subdir, file)
#                     data = loadmat(file_path)
#                     features = data['data']
#                     sentence_sizes.append(features.shape[0]) 
#                     label = self.get_label(subdir)
#                     self.data.append(features)
#                     self.labels.append(label)

#     def get_label(self, subdir):
#         if 'control' in subdir:
#             return 0
#         elif 'patient' in subdir:
#             return 1
#         elif 'prodromal' in subdir:
#             return 2
#         elif 'swedd' in subdir:
#             return 3
#         else:
#             print("Label error")
#             return -1  

#     def edge_attr_dim(self):
#         return 1     

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         data = self.data[idx]
#         label = self.labels[idx]
#         data = torch.corrcoef(torch.tensor(data).squeeze().T)
#         data = torch.nan_to_num(data)
#         data = pearson_dataset(data, label, 10)
#         return data
#         # data = torch.tensor(self.data[idx]).float()
#         # label = self.labels[idx]
        
#         # # Compute the correlation matrix
#         # corr_matrix = torch.corrcoef(data.squeeze().T)
#         # corr_matrix = torch.nan_to_num(corr_matrix)
        
#         # # Threshold the correlation matrix to create an adjacency matrix
#         # adjacency_matrix = preprocess_adjacency_matrix(corr_matrix, 10)
#         # # Since adjacency_matrix is a tuple, let's print its components
#         # edge_index, edge_attr = from_scipy_sparse_matrix(adjacency_matrix)

#         # # Now convert edge_index and edge_attr to graph
#         # G = nx.Graph(adjacency_matrix)

        
#         # # Compute eigenvector centrality
#         # eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
         
#         # # Create a feature vector where each node feature is its eigenvector centrality score
#         # node_features = torch.tensor([eigenvector_centrality[i] for i in range(len(eigenvector_centrality))]).float().unsqueeze(1)  # Shape [num_nodes, 1]
#         # # Add binary hemisphere feature: 0 for left hemisphere, 1 for right hemisphere
#         # num_nodes = len(eigenvector_centrality)
#         # hemisphere_feature = torch.zeros(num_nodes, 1)  # Initialize with 0 (left hemisphere)
#         # hemisphere_feature[1::2] = 1  # Set to 1 for right hemisphere (alternating L, R)
#         # # Concatenate eigenvector centrality and hemisphere features
#         # node_features = torch.cat([node_features, hemisphere_feature], dim=1)  # Shape [num_nodes, 2]
#         # # Create PyG data object with eigenvector centrality as node features
#         # data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        
#         return data


