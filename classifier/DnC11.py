import torch
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_score
from torch.nn import BatchNorm1d
import argparse
from torch_scatter import scatter


from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from pipelines.dataset import ConnectivityWCurvaturesDataset

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a GCN model for neurodegenerative disease classification.')
parser.add_argument('--dataset', type=str, default='ppmi', help='Dataset to use for training (ppmi, adni)')
parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--seed', type=int, default=52, help='Random seed for reproducibility')
parser.add_argument('--hidden_layer_size', type=int, default=512, help='Number of hidden units in each GCN layer')
parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers in the GCN')
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

out_channels = 4 if args.dataset == 'ppmi' else 2

dataset = ConnectivityWCurvaturesDataset(connectivities, curvature, labels)

class CustomGATConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        super(CustomGATConv, self).__init__(aggr='add')  # "Add" aggregation for attention
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # Linear transformation for node features
        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        # Linear transformation for edge features
        self.edge_lin = torch.nn.Linear(edge_dim, heads * out_channels, bias=False)
        # Attention parameters for each head
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, out_channels * 3))

        # Optional bias
        if bias and concat:
            self.bias = torch.nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.edge_lin.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin(x)  # Apply linear transformation to node features
        edge_attr = self.edge_lin(edge_attr)  # Apply linear transformation to edge features
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)  # Start the message passing

    def message(self, x_i, x_j, edge_attr, index, ptr, size_i):
        # Compute attention coefficients
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        print(f"x_i: {x_i.shape}, x_j: {x_j.shape}, edge_attr: {edge_attr.shape}")
        
        # Concatenate node features and edge features for attention mechanism
        combined_features = torch.cat([x_i, x_j, edge_attr], dim=-1)
        print(f"combined_features: {combined_features.shape}, att: {self.att.shape}")
        
        # Compute attention scores
        alpha = (combined_features * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = torch.nn.functional.dropout(alpha, p=self.dropout, training=self.training)
        alpha = F.softmax(alpha, dim=1)
        print(f"alpha: {alpha.shape}")        
        
        # Weight the messages by the attention coefficients
        weighted_messages = x_j * alpha.view(-1, self.heads, 1)
        print(f"weighted_messages: {weighted_messages.shape}")
        return weighted_messages

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Custom aggregation to ensure dimensions match
        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')

    def update(self, aggr_out):
        # aggr_out contains the aggregated messages for each node
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)
        
        # Optionally, add a bias term
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        return aggr_out  # Return the updated node embeddings

class GAT(torch.nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_layer_size, num_hidden_layers, num_classes, heads=1):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomGATConv(node_feature_dim, hidden_layer_size, edge_feature_dim, heads=heads))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(CustomGATConv(hidden_layer_size * heads, hidden_layer_size, edge_feature_dim, heads=heads))
        self.lin = torch.nn.Linear(hidden_layer_size * heads, num_classes)

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

    model = GAT(node_feature_dim, edge_feature_dim, args.hidden_layer_size, args.num_hidden_layers, out_channels, heads=1).to(device)
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
            print(f'output shape: {output.shape}, data.y shape: {data.y.shape}')
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
