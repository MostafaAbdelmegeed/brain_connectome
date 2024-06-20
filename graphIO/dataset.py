import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader


class BrainConnectivityDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.y = self.labels[idx]
        return data
    

def create_data_loaders(graphs, labels, n_splits=5, batch_size=32, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    loaders = []
    for train_index, test_index in skf.split(graphs, labels):
        train_graphs = [graphs[i] for i in train_index]
        train_labels = labels[train_index]
        test_graphs = [graphs[i] for i in test_index]
        test_labels = labels[test_index]
        train_dataset = BrainConnectivityDataset(train_graphs, train_labels)
        test_dataset = BrainConnectivityDataset(test_graphs, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        loaders.append((train_loader, test_loader))
    return loaders