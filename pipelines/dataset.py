import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from .functional_encoding import yeo_network

# def yeo_network(as_tensor=False):
#     # Define the refined functional classes based on Yeo's 7 Network Parcellations
#     visual = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
#     somatomotor = [0, 1, 57, 58, 69, 70, 16, 17, 18, 19]
#     dorsal_attention = [59, 60, 61, 62, 67, 68, 6, 7]
#     ventral_attention = [12, 13, 63, 64, 29, 30]
#     limbic = [37, 38, 39, 40, 41, 42, 31, 32, 33, 34, 35, 36]
#     frontoparietal = [2, 3, 4, 5, 6, 7, 10, 11]
#     default_mode = [23, 24, 67, 68, 65, 66, 35, 36]
#     # Initialize the one-hot encoding list
#     one_hot_encodings = []
#     # Create one-hot encodings for each region
#     for i in range(116):
#         encoding = [
#             1 if i in visual else 0,
#             1 if i in somatomotor else 0,
#             1 if i in dorsal_attention else 0,
#             1 if i in ventral_attention else 0,
#             1 if i in limbic else 0,
#             1 if i in frontoparietal else 0,
#             1 if i in default_mode else 0
#         ]
#         one_hot_encodings.append(encoding)
#     if as_tensor:
#         return torch.tensor(one_hot_encodings, dtype=torch.float)
#     else:
#         return one_hot_encodings


# Define the Dataset class
class ConnectivityDataset(Dataset):
    def __init__(self, connectivities, labels):
        self.connectivities = connectivities
        self.node_num = len(connectivities[0])
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()

    def get_hemisphere_indices(self):
        left_indices = [i for i in range(self.node_num) if i % 2 == 0]
        right_indices = [i for i in range(self.node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self,i,j):
        if i in self.left_indices and j in self.right_indices:
            return True
        if i in self.right_indices and j in self.left_indices:
            return True
        return False

    def isIntra(self,i,j):
        if i in self.left_indices and j in self.left_indices:
            return True
        if i in self.right_indices and j in self.right_indices:
            return True
        return False
    
    def isHomo(self,i,j):
        return i//2 == j//2 and abs(i-j) == 1

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        node_attr = yeo_network(as_tensor=False)

        for j in range(connectivity.shape[0]):
            for k in range(connectivity.shape[0]):
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isIntra(j, k), self.isHomo(j, k)])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        node_features = torch.tensor(np.array(node_attr), dtype=torch.float)
        y = label.type(torch.long)
        # print(f'node_features: {node_features.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}, y: {y}')
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    


class DivideAndConquerDataset(Dataset):
    def __init__(self, connectivities, labels):
        self.connectivities = connectivities
        self.node_num = len(connectivities[0])
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()


    def get_inter(self, idx):
        return self.connectivities[idx][self.left_indices][:, self.right_indices]
    
    def get_intra(self, idx, hemisphere='left'):
        return self.connectivities[idx][self.left_indices][:, self.left_indices] if hemisphere == 'left' else self.connectivities[idx][self.right_indices][:, self.right_indices]
    
    def get_homotopic(self, idx):
        return self.connectivities[idx][self.left_indices, self.right_indices]


    def get_hemisphere_indices(self):
        left_indices = [i for i in range(self.node_num) if i % 2 == 0]
        right_indices = [i for i in range(self.node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self,i,j):
        if i in self.left_indices and j in self.right_indices:
            return True
        if i in self.right_indices and j in self.left_indices:
            return True
        return False

    def isIntra(self,i,j):
        if i in self.left_indices and j in self.left_indices:
            return True
        if i in self.right_indices and j in self.right_indices:
            return True
        return False
    
    def isHomo(self,i,j):
        return i//2 == j//2 and abs(i-j) == 1

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]
        node_attr = yeo_network(as_tensor=False)[::2]
        edge_index = []
        edge_attr = []
        inter = self.get_inter(idx)
        intra_left = self.get_intra(idx, 'left')
        intra_right = self.get_intra(idx, 'right')
        homotopic = self.get_homotopic(idx)
        for j in range(connectivity.shape[0]//2):
            for k in range(connectivity.shape[0]//2):
                edge_index.append([j, k])
                edge_attr.append([inter[j, k], intra_left[j, k], intra_right[j, k], homotopic[j]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        node_features = torch.tensor(np.array(node_attr), dtype=torch.float)
        y = label.type(torch.long)
        # print(f'node_features: {node_features.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}, y: {y}')
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data
    