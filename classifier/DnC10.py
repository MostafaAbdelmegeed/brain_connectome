import torch
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score
from torch.nn import BatchNorm1d
import argparse

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing

from pipelines.dataset import ConnectivityDataset

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a GCN model for neurodegenerative disease classification.')
parser.add_argument('--dataset', type=str, default='ppmi', help='Dataset to use for training (ppmi, adni)')
parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--seed', type=int, default=52, help='Random seed for reproducibility')
parser.add_argument('--hidden_layer_size', type=int, default=512, help='Number of hidden units in each GCN layer')
parser.add_argument('--num_hidden_layers', type=int, default=4, help='Number of hidden layers in the GCN')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use for training')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
args = parser.parse_args()

print(f'Arguments: {args}')

# Load dataset
data = torch.load(f'data/{args.dataset}.pth')
curv = torch.load(f'data/{args.dataset}_curv.pth')
connectivities = data['matrix'].numpy()
curvature = curv['matrix'].numpy()
labels = data['label']

node_feature_dim = 7
edge_feature_dim = 6

dataset = ConnectivityDataset(connectivities, curvature, labels)

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

# Determine device
device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

# Cross-validation setup
kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
# Initialize lists to store metrics for all folds
all_fold_metrics = {'accuracy': [], 'precision': [], 'f1': []}

# Training and evaluation loop
for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}/{args.num_folds}")
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    # Split training data into train and validation sets for early stopping
    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=0.2, random_state=args.seed)
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = GCN(node_feature_dim, edge_feature_dim, args.hidden_layer_size, args.num_hidden_layers, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                val_loss += criterion(output, data.y).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Evaluate the model on the test set
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
    print(f"Fold {fold + 1} Metrics: Accuracy: {acc}, Precision: {precision}, F1 Score: {f1}")

    # Store metrics for the current fold
    all_fold_metrics['accuracy'].append(acc)
    all_fold_metrics['precision'].append(precision)
    all_fold_metrics['f1'].append(f1)

    # Print average metrics until the latest fold
    avg_accuracy = np.mean(all_fold_metrics['accuracy'])
    avg_precision = np.mean(all_fold_metrics['precision'])
    avg_f1 = np.mean(all_fold_metrics['f1'])
    print(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy}, Precision: {avg_precision}, F1 Score: {avg_f1}")

# Compute and print final metrics after cross-validation
final_accuracy = np.mean(all_fold_metrics['accuracy'])
final_precision = np.mean(all_fold_metrics['precision'])
final_f1 = np.mean(all_fold_metrics['f1'])

print("Training completed.")
print(f"Final Metrics:\nAccuracy: {final_accuracy}\nPrecision: {final_precision}\nF1 Score: {final_f1}")
