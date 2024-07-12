import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.preprocess import preprocess_adjacency_matrix
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

class Dataset_ADNI(Dataset):
    def __init__(self, data_dir, label_file, sparsity=10):
        super(Dataset_ADNI, self).__init__()
        self.data_dir = data_dir
        self.label_file = label_file
        self.sparsity = 10
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        sentence_sizes = []
        labels_df = pd.read_csv(self.label_file, header=0)
        for filename in os.listdir(self.data_dir):
            if filename.startswith('sub_'):
                id = filename.split('_')[1]
                label_row = labels_df[labels_df['subject_id'] == id]
                if not label_row.empty:
                    label = label_row.iloc[0]['DX']
                    if label in ['CN', 'SMC', 'EMCI']:
                        self.labels.append(0)
                    elif label in ['LMCI', 'AD']:
                        self.labels.append(1)
                    else:
                        print('Label Error')
                        self.labels.append(-1)
                    features = np.loadtxt(os.path.join(self.data_dir, filename))
                    self.data.append(features)
                    sentence_sizes.append(features.shape[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = self.labels[idx]
        data = torch.corrcoef(data.squeeze().T)
        adjacency_matrix = torch.nan_to_num(data)
        adjacency_matrix[adjacency_matrix < np.percentile(adjacency_matrix.flatten(), 100-self.sparsity)] = 0
        # TODO: Work on adjacency matrix to get subgraphs
        data_adj = from_scipy_sparse_matrix(sp.coo_matrix(adjacency_matrix))
        data = Data(x=data.float(), edge_index=data_adj[0], edge_attr=data_adj[1], y=torch.tensor(label))
        return data

class Dataset_PPMI(Dataset):
    def __init__(self, root_dir, sparsity=10):
        super(Dataset_PPMI, self).__init__()
        self.root_dir = root_dir
        self.sparsity = sparsity
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
        
    def pad_sentences(self):
        self.data = [torch.cat((torch.tensor(sentence), torch.zeros(self.max_sentence_size - sentence.shape[0], sentence.shape[1])), dim=0) for sentence in self.data]        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx]).float()
        label = self.labels[idx]
        data = torch.corrcoef(data.squeeze().T)
        data = torch.nan_to_num(data)
        adjacency_matrix = preprocess_adjacency_matrix(data, 10)
        data = Data(x=data.float(), edge_index=adjacency_matrix[0], edge_attr=adjacency_matrix[1], y=torch.tensor(label))
        return data