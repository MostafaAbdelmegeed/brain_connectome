import torch
from torch_geometric.data import Data, Dataset, Batch
import os
from scipy.io import loadmat
from torch.utils.data import Dataset as TorchDataset
from preprocessing import *
import networkx as nx
import pandas as pd


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
            labels_id = {'CN': 0, 'SMC': 0, 'EMCI': 1, 'LMCI': 2, 'AD': 3}
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

    

