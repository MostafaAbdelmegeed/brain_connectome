from torch_geometric.nn import GATConv, global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch
import sys
from pathlib import Path
import numpy as np

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from graphIO.dataset import BrainConnectivityDataset
from graphIO.io import load_data
from graphIO.preprocess import standardize_matrices, construct_graph, convert_to_pyg_data
from graphIO.graphIO_dep import read_adj_matrices_from_directory
import argparse
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import time

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        # Input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=0.6))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads, dropout=0.6))
        # Output layer
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6))
        # Linear layers
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.dropout(x, p=0.6, training=self.training)
            x = F.elu(conv(x, edge_index))

        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

def create_data_loaders(graphs, labels, n_splits=5, batch_size=32):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
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

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for data in tqdm(train_loader, desc='Training', leave=False):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))  # Flatten the target labels
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        # Calculate accuracy
        pred = out.argmax(dim=1)
        correct += pred.eq(data.y.view(-1)).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', leave=False):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y.view(-1))  # Flatten the target labels
            total_loss += loss.item() * data.num_graphs
            # Calculate accuracy
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y.view(-1)).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

# python classifier/baseline.py --dataset_path data/ppmi_corr_116.pth --in_channels 116 --hidden_channels 16 --out_channels 4 --heads 2 --num_layers 2 --device cuda --epochs 20 --learning_rate 0.01 --suppress_threshold 0.3 --folds 1 --batch_size 1

if __name__ == '__main__':
    # Step 1: Import argparse
    parser = argparse.ArgumentParser(description='Baseline GAT Model Training')
    # Step 2: Define command-line arguments
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--in_channels', type=int, required=True, help='Number of input channels')
    parser.add_argument('--hidden_channels', type=int, required=True, help='Number of hidden channels')
    parser.add_argument('--out_channels', type=int, required=True, help='Number of output channels')
    parser.add_argument('--heads', type=int, required=True, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, required=True, help='Number of GAT layers')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--suppress_threshold', type=float, default=0.3, help='Threshold for suppressing connections')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    # Step 3: Parse the arguments
    args = parser.parse_args()
    # Step 4: Set the device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("CUDA available:", torch.cuda.is_available())
    print(f"Using device: {device}")
    print("Loading dataset...")
    ppmi_dataset = torch.load(args.dataset_path)
    print("Standardizing matrices...")
    # Extract data and labels from the loaded dataset
    connectivity_matrices = standardize_matrices(ppmi_dataset['data'].numpy())
    connectivity_labels = ppmi_dataset['class_label']
    # Filter matrices and labels to keep only control (0) and patient (2) classes
    mask = (connectivity_labels == 0) | (connectivity_labels == 2)
    connectivity_matrices = connectivity_matrices[mask]
    connectivity_labels = connectivity_labels[mask].numpy()
    connectivity_labels[connectivity_labels == 2] = 1
    print(f'Original matrices shape: {connectivity_matrices.shape}')
    print(f'Original labels shape: {connectivity_labels.shape}')
    graphs = []
    print("Constructing graphs...")
    for i in range(len(connectivity_matrices)):
        graphs.append(construct_graph(connectivity_matrices[i], args.suppress_threshold))
    print("Converting graphs to PyTorch Geometric format...")
    graphs = [convert_to_pyg_data(G, args.in_channels) for G in graphs]
    labels = torch.tensor(connectivity_labels, dtype=torch.long)
    print("Creating data loaders...")
    loaders = create_data_loaders(graphs, labels, n_splits=args.folds, batch_size=args.batch_size)
    print("Data loaders created.")
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'runs/baseline_in{args.in_channels}_h{args.hidden_channels}_out{args.out_channels}_a{args.heads}.pth')
    print("Starting training with cross-validation...")
    # Training with Cross-Validation
    for fold, (train_loader, test_loader) in enumerate(loaders):
        print(f"Training fold {fold + 1}...")
        model = GAT(args.in_channels, args.hidden_channels, args.out_channels, args.heads, args.num_layers).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        best_model_wts = None
        for epoch in range(args.epochs):  # Adjust the number of epochs as needed
            start_time = time.time()
            train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = test_model(model, test_loader, criterion, device)
            elapsed_time = time.time() - start_time
            # Log metrics to TensorBoard
            writer.add_scalar(f'Fold_{fold+1}/Train_Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Train_Acc', train_acc, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Test_Loss', test_loss, epoch)
            writer.add_scalar(f'Fold_{fold+1}/Test_Acc', test_acc, epoch)
            print(f'[Fold {fold + 1}][Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {elapsed_time:.2f}s')
            # Save the best model
            if test_acc > best_acc:
                best_acc = test_acc
                best_model_wts = model.state_dict()
        # Save the best model for this fold
        model_save_path = f'models/baseline_in{args.in_channels}_h{args.hidden_channels}_out{args.out_channels}_a{args.heads}_layers{args.num_layers}.pth'
        torch.save({
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, model_save_path)
        print(f'Saved best model of fold {fold + 1} to {model_save_path}')
    # Close the TensorBoard writer
    writer.close()
    print("Training finished.")
