import torch
from torch_geometric.data import Data, Dataset, Batch
import os
from scipy.io import loadmat
from torch.utils.data import Dataset as TorchDataset
from help_funs import *
import networkx as nx
import pandas as pd


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
    
class PPMIStructuralEdgesDataset(Dataset):
    def __init__(self, root_dir, mgnn):
        super(PPMIStructuralEdgesDataset, self).__init__()
        self.root_dir = root_dir
        self.data = []
        self.labels = []
        self.load_data()
        self.mgnn = mgnn

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
        data = torch.corrcoef(torch.tensor(data).squeeze().T)#.fill_diagonal_(0)
        data = torch.nan_to_num(data)
        data = pearson_dataset(data, label, 10, self.mgnn)
        data.edge_attr = add_structural_features(data.edge_index, data.edge_attr)
        return data  

class PPMIAsymmetryDataset(Dataset):
    def __init__(self, root_dir):
        super(PPMIAsymmetryDataset, self).__init__()
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
        edge_index, edge_attr = preprocess_adjacency_matrix(data, 10)
        data = torch.abs(data[:, 1::2] - data[:, 0::2])
        data = Data(x=data.float(), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
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

class PPMISimpleDataset(Dataset):
    def __init__(self, root_dir):
        super(PPMISimpleDataset, self).__init__()
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
        data = pearson_dataset(data, label, 10)
        data.edge_attr = add_structural_features(data.edge_index, data.edge_attr)
        data.x = data.x.mean(dim=1).unsqueeze(1)
        return data

class BrainDataset(Dataset):
    def __init__(self, data):
        self.connectivities = data['connectivity']
        self.labels = data['label']
        self.node_num = self.connectivities[0].shape[0]

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]

        # # Ensure coalesced
        # edge_adj = edge_adj.coalesce()
        # num_nodes = connectivity.shape[0]

        # Initialize lists for edge_index and edge_attr
        # edge_index = []
        # edge_attr = []

        # Initialize node degree, strength, and hemisphere indicator tensors
        # node_degree = torch.zeros(num_nodes, dtype=torch.float)
        # node_strength = torch.zeros(num_nodes, dtype=torch.float)
        # hemisphere_indicator = torch.zeros(num_nodes, dtype=torch.float)
        # hemisphere_indicator[[i for i in range(self.node_num) if i % 2 != 0]] = 1.0  # Right hemisphere nodes get 1.0
        # print(f'Connectivity min: {connectivity.min()}, max: {connectivity.max()}')
        # Populate edge_index, edge_attr, and compute node features
        edge_index, edge_attr = from_scipy_sparse_matrix(sp.coo_matrix(connectivity))
        edge_attr = add_structural_features(edge_index, edge_attr)
        # Separate continuous and binary features
        # node_features = torch.stack([
        #     node_degree,
        #     node_strength,
        #     node_clustering
        # ], dim=1)

        # Convert lists to tensors
        # edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        # edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(
            x=connectivity,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label.type(torch.long)
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


class Dataset_PPMI(TorchDataset):
    def __init__(self, root_dir, mgnn=False):
        super(Dataset_PPMI, self).__init__()
        self.root_dir = root_dir
        self.mgnn = mgnn
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
        data = pearson_dataset(data, label, 10, self.mgnn)
        return data
    

class Dataset_ADNI(TorchDataset):
    def __init__(self, data_dir, label_file, num_classes=2):
        super(Dataset_ADNI, self).__init__()
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.label_file = label_file
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        labels_df = pd.read_csv(self.label_file, header=0)
        for filename in os.listdir(self.data_dir):
            sentence_sizes = []
        labels_df = pd.read_csv(self.label_file, header=0)
        if self.num_classes == 4:
            labels_id = {'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 2, 'AD': 3}
        elif self.num_classes == 2:
            labels_id = {'CN': 0, 'SMC': 0, 'EMCI': 0, 'LMCI': 1, 'AD': 1}
        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]
                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    self.labels.append(labels_id[label])
                    features = np.loadtxt(os.path.join(self.data_dir, filename))
                    self.data.append(features)
                    sentence_sizes.append(features.shape[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = self.labels[idx]
        data = torch.corrcoef(data.squeeze().T)  
        data = torch.nan_to_num(data)
        data = pearson_dataset(data, label, 10)
        return data
    


class ADNIAsymmetryDataset(TorchDataset):
    def __init__(self, data_dir, label_file, num_classes=2, mgnn=False):
            super(ADNIAsymmetryDataset, self).__init__()
            self.mgnn = mgnn
            self.data_dir = data_dir
            self.label_file = label_file
            self.num_classes = num_classes
            self.data = []
            self.labels = []
            self.load_data()

    def load_data(self):
        sentence_sizes = []
        labels_df = pd.read_csv(self.label_file, header=0)
        if self.num_classes == 4:
            labels_id = {'CN': 0, 'SMC': 1, 'EMCI': 2, 'LMCI': 2, 'AD': 3}
        elif self.num_classes == 2:
            labels_id = {'CN': 0, 'SMC': 0, 'EMCI': 0, 'LMCI': 1, 'AD': 1}
        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]
                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    self.labels.append(labels_id[label])
                    features = np.loadtxt(os.path.join(self.data_dir, filename))
                    self.data.append(features)
                    sentence_sizes.append(features.shape[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.corrcoef(torch.tensor(data).squeeze().T)
        data = torch.nan_to_num(data)
        edge_index, edge_attr = preprocess_adjacency_matrix(data, 10)
        data = torch.abs(data[:, 1::2] - data[:, 0::2])
        data = Data(x=data.float(), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        return data 
    

    

