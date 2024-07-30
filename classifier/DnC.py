import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a GCN model for neurodegenerative disease classification.')
parser.add_argument('--dataset', type=str, default='ppmi', help='Dataset to use for training (ppmi, adni)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--hidden_layer_size', type=int, default=512, help='Number of hidden units in each GCN layer')
parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in the GCN')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
args = parser.parse_args()

# Set hyperparameters from command-line arguments
dataset_name = args.dataset
batch_size = args.batch_size
hidden_layer_size = args.hidden_layer_size
num_hidden_layers = args.num_hidden_layers
epochs = args.epochs
learning_rate = args.learning_rate
num_folds = args.num_folds
gpu_id = args.gpu_id

num_classes = 4 if dataset_name == 'ppmi' else 2
node_feature_dim = 116
edge_feature_dim = 4  # Updated to 4 to include the original connectivity values

# Check if GPU is available and set device
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load data
dataset = torch.load(f'data/{dataset_name}.pth')
connectivities = dataset['matrix'].numpy()
labels = dataset['label']

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

        inter_matrix = self.extract_interhemispheric_difference(connectivity)
        intra_asym_matrix = self.extract_intrahemispheric_asymmetry_matrix(connectivity)
        homotopic_matrix = self.extract_homotopic_difference(connectivity)
        combined_matrix = self.combine_feature_matrices(connectivity, inter_matrix, intra_asym_matrix, homotopic_matrix)

        edge_index = []
        edge_attr = []
        for j in range(node_feature_dim):
            for k in range(node_feature_dim):
                if np.any(combined_matrix[j, k] != 0):
                    edge_index.append([j, k])
                    edge_attr.append(combined_matrix[j, k])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)

        # Normalize edge attributes
        edge_attr = (edge_attr - edge_attr.mean(dim=0, keepdim=True)) / (edge_attr.std(dim=0, keepdim=True) + 1e-6)

        # Create node features with the same integer for corresponding regions in both hemispheres
        node_features = torch.zeros(node_feature_dim, dtype=torch.float).unsqueeze(1)
        for i in range(len(self.left_indices)):
            node_features[self.left_indices[i]] = i
            node_features[self.right_indices[i]] = i

        # Normalize node features
        node_features = (node_features - node_features.mean()) / (node_features.std() + 1e-6)

        y = label.clone().detach()  # Correctly define the label tensor

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def extract_interhemispheric_difference(self, connectivity):
        interhemispheric_difference = np.zeros((58, 58))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                interhemispheric_difference[i, j] = connectivity[li, ri] - connectivity[ri, li]
        return interhemispheric_difference

    def extract_intrahemispheric_asymmetry_matrix(self, connectivity):
        intra_asymmetry_matrix = np.zeros((node_feature_dim, node_feature_dim))
        left_hemisphere = connectivity[np.ix_(self.left_indices, self.left_indices)]
        right_hemisphere = connectivity[np.ix_(self.right_indices, self.right_indices)]
        intra_asymmetry_matrix[np.ix_(self.left_indices, self.left_indices)] = np.abs(left_hemisphere - left_hemisphere.T)
        intra_asymmetry_matrix[np.ix_(self.right_indices, self.right_indices)] = np.abs(right_hemisphere - right_hemisphere.T)
        return intra_asymmetry_matrix

    def extract_homotopic_difference(self, connectivity):
        homotopic_difference = np.zeros((58, 58))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                homotopic_difference[i, j] = connectivity[li, ri] - connectivity[ri, li]
        return homotopic_difference

    def combine_feature_matrices(self, connectivity, inter_matrix, intra_asym_matrix, homotopic_matrix):
        combined_matrix = np.zeros((node_feature_dim, node_feature_dim, edge_feature_dim))
        for i, li in enumerate(self.left_indices):
            for j, ri in enumerate(self.right_indices):
                combined_matrix[li, ri, 0] = connectivity[li, ri]  # Original connectivity
                combined_matrix[li, ri, 1] = inter_matrix[i, j]
                combined_matrix[li, ri, 3] = homotopic_matrix[i, j]
        combined_matrix[:, :, 2] = intra_asym_matrix
        return combined_matrix

class EdgeEnhancedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeEnhancedGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels + edge_feature_dim, out_channels)  # Include edge features.
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout for regularization

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
        return self.dropout(self.lin(aggr_out))  # Linearly transform and apply dropout to the aggregated messages

class GCN(torch.nn.Module):
    def __init__(self, node_in_channels, hidden_channels, out_channels, num_hidden_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(EdgeEnhancedGCNConv(node_in_channels, hidden_channels))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(EdgeEnhancedGCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout for regularization

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device), data.batch.to(device)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_mean_pool(x, batch)  # Global pooling for graph-level classification
        x = self.dropout(x)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

def train_gcn(train_loader, val_loader, epochs=200, lr=0.01):
    model = GCN(node_in_channels=1, hidden_channels=hidden_layer_size, out_channels=num_classes, num_hidden_layers=num_hidden_layers).to(device)  # Updated input channel size to 1
    print(f'Model Parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, data.y.to(device))
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'Epoch {epoch+1}, Loss is NaN or Inf, stopping training.')
                return
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        
        # Validation step
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in val_loader:
                out = model(data)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

def cross_validation(dataset, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold = 1
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    for train_index, val_index in kf.split(dataset):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        print(f'Fold {fold}')
        accuracy, precision, recall, f1 = train_gcn(train_loader, val_loader, epochs=epochs, lr=learning_rate)
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # Print average metrics after each fold
        avg_accuracy = np.mean(all_accuracies)
        avg_precision = np.mean(all_precisions)
        avg_recall = np.mean(all_recalls)
        avg_f1 = np.mean(all_f1s)
        print(f'Average after Fold {fold} -> Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')
        
        fold += 1

    # Calculate and print overall metrics
    avg_accuracy = np.mean(all_accuracies)
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)
    print(f'\nOverall Metrics -> Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}')

dataset = ConnectivityDataset(connectivities, labels)
cross_validation(dataset, num_folds=num_folds)
