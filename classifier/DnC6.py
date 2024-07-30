import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import global_mean_pool, GCNConv
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
node_num = 116
node_feature_dim = 4
edge_feature_dim = 4  # Updated to 4 to include the original connectivity values

# Check if GPU is available and set device
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

# Load data
dataset = torch.load(f'data/{dataset_name}_curv.pth')
connectivities = dataset['matrix'].numpy()
labels = dataset['label']

# Functional Classification of AAL116 Regions
motor_regions = [1, 2, 3, 4, 5, 6, 9, 10, 19, 20, 21, 22]
cerebellum_regions = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
cognitive_regions = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 49, 50, 51, 52, 53, 54]
limbic_regions = [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]

# Combine relevant regions based on the dataset
if dataset_name == 'ppmi':
    focus_regions = motor_regions + cerebellum_regions
else:
    focus_regions = cognitive_regions + limbic_regions


# Define the Dataset class
class ConnectivityDataset(Dataset):
    def __init__(self, connectivities, labels):
        self.connectivities = connectivities
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()

    def get_hemisphere_indices(self):
        left_indices = [i for i in range(node_num) if i % 2 == 0]
        right_indices = [i for i in range(node_num) if i % 2 != 0]
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
        node_attr = []
        for j in range(node_num):
            node_attr.append([j in motor_regions, j in cerebellum_regions, j in cognitive_regions, j in limbic_regions])
            for k in range(node_num):
                edge_index.append([j, k])
                edge_attr.append([connectivity[j,k], self.isInter(j,k), self.isIntra(j,k), self.isHomo(j,k)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        node_features = torch.tensor(node_attr, dtype=torch.float)
        y = label.type(torch.long)
        # print(f'edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}, node_features: {node_features.shape}, label: {y.shape}')
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data


# Custom GCN layer to incorporate edge features
class CustomGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(CustomGCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.linear = torch.nn.Linear(in_channels * 2 + edge_dim, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Apply linear transformation to node and edge features
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate node and edge features
        # print(f'x_i: {x_i.shape}, x_j: {x_j.shape}, edge_attr: {edge_attr.shape}')
        edge_features = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.linear(edge_features)

    def update(self, aggr_out):
        return aggr_out

# Custom GCN Model
class GCN(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_layer_size, num_hidden_layers, num_classes):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomGCNConv(node_feature_dim, hidden_layer_size, edge_feature_dim))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(CustomGCNConv(hidden_layer_size, hidden_layer_size, edge_feature_dim))
        self.lin = torch.nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# Instantiate model, loss function, and optimizer
model = GCN(node_feature_dim, edge_feature_dim, hidden_layer_size, num_hidden_layers, num_classes).to(device)
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