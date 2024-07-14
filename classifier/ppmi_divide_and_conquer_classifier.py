import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import KFold
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a GCN model for Parkinson\'s disease classification.')
parser.add_argument('--dataset', type=str, default='PPMI', help='Which dataset to use (PPMI or ADNI)', choices=['PPMI', 'ADNI'])
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--hidden_layer_size', type=int, default=512, help='Number of hidden units in each GCN layer')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation')
args = parser.parse_args()

# Set hyperparameters from command-line arguments
batch_size = args.batch_size
hidden_layer_size = args.hidden_layer_size
epochs = args.epochs
learning_rate = args.learning_rate
num_folds = args.num_folds
node_feature_dim = 116
edge_feature_dim = 3
num_classes = 4

# Load data
connectivity_dataset = torch.load('data/ppmi.pth') if args.dataset == 'PPMI' else torch.load('data/adni.pth')
connectivities = connectivity_dataset['matrix'].numpy()
labels = connectivity_dataset['label']
print(f'matrices: {connectivities.shape}, labels: {labels.shape}')
print(f'Connectivity matrices shape: {connectivities.shape}, Labels shape: {labels.shape}')
print(f'Connectivity matrices dtype: {connectivities.dtype}, Labels dtype: {labels.dtype}')
print(f'Connectivity matrices min: {connectivities.min()}, Labels min: {labels.min()}')
print(f'Connectivity matrices max: {connectivities.max()}, Labels max: {labels.max()}')

class ConnectivityDataset(Dataset):
    def __init__(self, connectivities, labels):
        """
        Args:
            connectivities (numpy.ndarray): Array of connectivity matrices of shape [n_samples, 116, 116].
            labels (torch.Tensor): Tensor of labels of shape [n_samples,].
        """
        self.connectivities = connectivities
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()

    def get_hemisphere_indices(self):
        left_indices = [i for i in range(node_feature_dim) if i % 2 == 0]
        right_indices = [i for i in range(node_feature_dim) if i % 2 != 0]
        return left_indices, right_indices

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]

        inter_matrix = self.extract_interhemispherical_matrix(connectivity)
        intra_asym_matrix = self.extract_intrahemispherical_asymmetry_matrix(connectivity)
        homotopic_matrix = self.extract_homotopic_matrix(connectivity)
        combined_matrix = self.combine_feature_matrices(inter_matrix, intra_asym_matrix, homotopic_matrix)

        edge_index = []
        edge_attr = []
        for j in range(node_feature_dim):
            for k in range(node_feature_dim):
                if np.any(combined_matrix[j, k] != 0):
                    edge_index.append([j, k])
                    edge_attr.append(combined_matrix[j, k])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        x = torch.tensor(np.eye(node_feature_dim), dtype=torch.float)  # Node features as identity matrix
        y = label.clone().detach()  # Correctly define the label tensor as a single long tensor

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def extract_interhemispherical_matrix(self, connectivity):
        interhemispherical_matrix = np.zeros((58, 58))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                interhemispherical_matrix[i, j] = connectivity[li, ri]
        return interhemispherical_matrix

    def extract_intrahemispherical_asymmetry_matrix(self, connectivity):
        intra_asymmetry_matrix = np.zeros((node_feature_dim, node_feature_dim))
        left_hemisphere = connectivity[np.ix_(self.left_indices, self.left_indices)]
        right_hemisphere = connectivity[np.ix_(self.right_indices, self.right_indices)]
        intra_asymmetry_matrix[np.ix_(self.left_indices, self.left_indices)] = np.abs(left_hemisphere - left_hemisphere.T)
        intra_asymmetry_matrix[np.ix_(self.right_indices, self.right_indices)] = np.abs(right_hemisphere - right_hemisphere.T)
        return intra_asymmetry_matrix

    def extract_homotopic_matrix(self, connectivity):
        homotopic_matrix = np.zeros((58, 58))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                homotopic_matrix[i, j] = connectivity[li, ri]
        return homotopic_matrix

    def combine_feature_matrices(self, inter_matrix, intra_asym_matrix, homotopic_matrix):
        combined_matrix = np.zeros((node_feature_dim, node_feature_dim, edge_feature_dim))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                combined_matrix[li, ri, 0] = inter_matrix[i, j]
                combined_matrix[li, ri, 2] = homotopic_matrix[i, j]
        combined_matrix[:, :, 1] = intra_asym_matrix
        return combined_matrix

class EdgeEnhancedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeEnhancedGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels + edge_feature_dim, out_channels)  # Include edge features.

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_feature_dim]

        # Step 1: Add self-loops to the adjacency matrix.
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Step 2: Add corresponding self-loop edge attributes.
        self_loop_attr = torch.zeros((num_nodes, edge_feature_dim), device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)

        # Step 3: Start propagating messages.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]
        # edge_attr has shape [E, edge_feature_dim]
        return torch.cat([x_j, edge_attr], dim=-1)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return self.lin(aggr_out)  # Linearly transform the aggregated messages

class GCN(torch.nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = EdgeEnhancedGCNConv(node_in_channels, hidden_channels)
        self.conv2 = EdgeEnhancedGCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)  # Global pooling for graph-level classification
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train_gcn(train_loader, val_loader, epochs=200, lr=0.01):
    model = GCN(node_in_channels=node_feature_dim, hidden_channels=hidden_layer_size, out_channels=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        # Validation step
        model.eval()
        correct = 0
        for data in val_loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
        accuracy = correct / len(val_loader.dataset)
        print(f'Validation Accuracy: {accuracy:.4f}')

def cross_validation(dataset, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold = 1
    for train_index, val_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f'Fold {fold}')
        train_gcn(train_loader, val_loader, epochs=epochs, lr=learning_rate)
        fold += 1

dataset = ConnectivityDataset(connectivities, labels)
cross_validation(dataset, num_folds=num_folds)
