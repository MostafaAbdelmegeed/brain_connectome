import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import global_mean_pool, GCNConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a GAT model for neurodegenerative disease classification.')
parser.add_argument('--dataset', type=str, default='ppmi', help='Dataset to use for training (ppmi, adni)')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--hidden_layer_size', type=int, default=512, help='Number of hidden units in each GAT layer')
parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in the GAT')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--num_folds', type=int, default=10, help='Number of folds for cross-validation')
parser.add_argument('--k_order', type=int, default=3, help='Order of Chebyshev polynomials')
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
K = args.k_order

num_classes = 4
node_num = 116
node_feature_dim = 9
edge_feature_dim = 4  # Updated to 4 to include the original connectivity values

# Check if GPU is available and set device
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load data
dataset = torch.load(f'data/{dataset_name if dataset_name == "ppmi" else "adni_multi"}.pth')
connectivities = dataset['matrix'].numpy()
labels = dataset['label']

from collections import defaultdict

# Initialize node to group mapping with multiple group support
node_to_groups = defaultdict(list)

# Define the functional groups with the full set of regions
motor_regions = [1, 2, 3, 4, 5, 6, 9, 10, 15, 16, 19, 20]
cerebellum_regions = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
cognitive_regions = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 49, 50, 51, 52]
limbic_regions = [57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
visual_regions = [43, 44, 45, 46, 47, 48]
auditory_regions = [41, 42]
dmn_regions = [23, 24, 25, 26, 55, 56, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78]
subcortical_regions = [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
somatosensory_regions = [7, 8, 11, 12, 13, 14, 17, 18, 19, 20]

# Update the mapping
for region in motor_regions:
    node_to_groups[region].append('motor')
for region in cerebellum_regions:
    node_to_groups[region].append('cerebellum')
for region in cognitive_regions:
    node_to_groups[region].append('cognitive')
for region in limbic_regions:
    node_to_groups[region].append('limbic')
for region in visual_regions:
    node_to_groups[region].append('visual')
for region in auditory_regions:
    node_to_groups[region].append('auditory')
for region in dmn_regions:
    node_to_groups[region].append('dmn')
for region in subcortical_regions:
    node_to_groups[region].append('subcortical')
for region in somatosensory_regions:
    node_to_groups[region].append('somatosensory')

# Define the list of functional groups
functional_groups = ['motor', 'cerebellum', 'cognitive', 'limbic', 'visual', 'auditory', 'dmn', 'subcortical', 'somatosensory']


def get_node_features(node_index, node_to_groups, functional_groups):
    features = np.zeros(len(functional_groups))
    groups = node_to_groups.get(node_index + 1, [])  # Add 1 because node indices start at 0
    for group in groups:
        features[functional_groups.index(group)] = 1
    return features




class CustomChebConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, edge_dim):
        super(CustomChebConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.edge_dim = edge_dim

        self.weight = torch.nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.edge_weight = torch.nn.Parameter(torch.Tensor(K, edge_dim, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.edge_weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr):
        deg = degree(edge_index[0], x.size(0), dtype=x.dtype)
        norm = deg.pow(-0.5)
        norm[torch.isinf(norm)] = 0
        norm = norm[edge_index[0]] * norm[edge_index[1]]

        Tx_0 = x
        Tx_1 = x

        out = torch.matmul(Tx_0, self.weight[0])
        if self.K > 1:
            out = out + torch.matmul(Tx_1, self.weight[1])

        for k in range(2, self.K):
            Tx_2 = 2 * self.propagate(edge_index, x=Tx_1, norm=norm) - Tx_0
            out = out + torch.matmul(Tx_2, self.weight[k])
            Tx_0, Tx_1 = Tx_1, Tx_2

        out += self.bias

        # Incorporate edge features
        edge_out = torch.matmul(edge_attr, self.edge_weight[0])
        for k in range(1, self.K):
            edge_out += torch.matmul(edge_attr, self.edge_weight[k])

        # Aggregate edge features to node features
        row, col = edge_index
        edge_out_aggregated = torch.zeros_like(out)
        edge_out_aggregated = edge_out_aggregated.index_add_(0, row, edge_out)

        return out + edge_out_aggregated

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GSN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_layer_size, num_hidden_layers, num_classes, K):
        super(GSN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomChebConv(node_feature_dim, hidden_layer_size, K, edge_feature_dim))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(CustomChebConv(hidden_layer_size, hidden_layer_size, K, edge_feature_dim))
        self.lin = torch.nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Update the Dataset class to include edge weights (which can be derived from the edge attributes)
class ConnectivityDataset(Dataset):
    def __init__(self, connectivities, labels):
        self.connectivities = connectivities
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()

    def get_hemisphere_indices(self):
        left_indices = [i for i in range(node_num) if i % 2 == 0]
        right_indices = [i for i in range(node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self, i, j):
        return (i in self.left_indices and j in self.right_indices) or (i in self.right_indices and j in self.left_indices)

    def isIntra(self, i, j):
        return (i in self.left_indices and j in  self.left_indices) or (i in  self.right_indices and j in  self.right_indices)

    def isHomo(self, i, j):
        return i // 2 == j // 2 and abs(i - j) == 1

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        node_attr = []

        for j in range(connectivity.shape[0]):
            node_attr.append(get_node_features(j, node_to_groups, functional_groups))
            for k in range(connectivity.shape[0]):
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isIntra(j, k), self.isHomo(j, k)])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        node_features = torch.tensor(np.array(node_attr), dtype=torch.float)
        edge_weight = edge_attr[:, 0]  # Use the connectivity values as edge weights
        y = label.type(torch.long)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_weight=edge_weight, y=y)
        return data

# Instantiate model, loss function, and optimizer
model = GSN(node_feature_dim, edge_feature_dim, hidden_layer_size, num_hidden_layers, num_classes, K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Cross-validation setup
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
dataset = ConnectivityDataset(connectivities, labels)

# Initialize lists to store metrics for all folds
all_fold_metrics = {'accuracy': [], 'precision': [], 'f1': []}

# Training and evaluation loop
for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{num_folds}")
    train_dataset = torch.utils.data.Subset(dataset, train_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        model.eval()
        preds = []
        labels = []
        for data in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = output.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels.extend(data.y.cpu().numpy())

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted')
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {acc}, Precision: {precision}, F1 Score: {f1}")

    # Store metrics for the current fold
    all_fold_metrics['accuracy'].append(acc)
    all_fold_metrics['precision'].append(precision)
    all_fold_metrics['f1'].append(f1)

# Compute and print final metrics after cross-validation
final_accuracy = np.mean(all_fold_metrics['accuracy'])
final_precision = np.mean(all_fold_metrics['precision'])
final_f1 = np.mean(all_fold_metrics['f1'])

print("Training completed.")
print(f"Final Metrics:\nAccuracy: {final_accuracy}\nPrecision: {final_precision}\nF1 Score: {final_f1}")