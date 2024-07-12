import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset, Subset
import os
from torch_geometric.nn import GCNConv, GINConv, GATConv, global_mean_pool, global_add_pool
from torch.nn import Linear
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm


# python classifier/GNNs_HCP.py --gpu 0 --model GCN --hidden_channels 8 --epochs 200 --input_path data/adni_raw.pth --batch_size 1 --lr 0.001 --num_layers 2 --heads 2 --in_feats 116 --out_feats 4


# Check if running in a non-interactive environment (e.g., using nohup)
is_non_interactive = not os.isatty(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'GAT'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--in_feats', type=int, default=116)
    parser.add_argument('--out_feats', type=int, default=4)
    parser.add_argument('--input_path', type=str, default='data/ppmi_raw.pth')
    parser.add_argument('--hidden_channels', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_splits', type=int, default=5)  # Number of cross-validation splits
    args = parser.parse_args()
    return args

def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(gpu_id):
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

def load_data(path='data/ppmi_raw.pth'):
    dataset = torch.load(path)
    return dataset

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lin1 = Linear(hidden_size, in_feats)
        self.lin2 = Linear(in_feats, out_feats)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x)
    
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

def get_model(model_name, hidden_channels, in_feats=116, out_feats=4, num_layers=2, heads=2):
    if model_name == 'GCN':
        return GCN(in_feats=in_feats, hidden_size=hidden_channels, out_feats=out_feats)
    elif model_name == 'GIN':
        return GIN(in_channels=in_feats, hidden_channels=hidden_channels, out_channels=out_feats, num_layers=num_layers)
    elif model_name == 'GAT':
        return GAT(in_channels=in_feats, hidden_channels=hidden_channels, out_channels=out_feats, heads=heads)
    else:
        raise argparse.ArgumentTypeError(f"Unsupported model. Choose from GCN, GIN, GAT.")

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data in tqdm(loader, desc="Training Batches", leave=False, dynamic_ncols=True, disable=is_non_interactive):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def test(model, loader, device):
    model.eval()
    correct = 0
    preds = []
    gts = []
    with torch.no_grad():
        for data in tqdm(loader, desc="Testing Batches", leave=False, dynamic_ncols=True, disable=is_non_interactive):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=-1)
            correct += int((pred == data.y).sum())
            preds.append(pred.cpu().numpy())
            gts.append(data.y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    accuracy = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average='weighted', zero_division=0)
    recall = recall_score(gts, preds, average='weighted', zero_division=0)
    f1 = f1_score(gts, preds, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def main():
    args = parse_args()
    set_seeds()
    device = get_device(args.gpu)
    dataset = load_data(path=args.input_path)
    
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=0)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}/{args.n_splits}')
        
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        
        model = get_model(args.model, args.hidden_channels, args.in_feats, args.out_feats, args.num_layers, args.heads).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        for epoch in tqdm(range(1, args.epochs + 1), desc="Training Epochs", leave=False, dynamic_ncols=True, disable=is_non_interactive):
            loss = train(model, train_loader, criterion, optimizer, device)
            val_acc, _, _, _ = test(model, test_loader, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        test_acc, pre, rec, f1 = test(model, test_loader, device)
        print(f'Fold {fold + 1} Results: Test Acc: {test_acc:.4f}, pre: {pre:.4f}, rec: {rec:.4f}, f1: {f1:.4f}')
        fold_results.append((test_acc, pre, rec, f1))
    
    avg_results = np.mean(fold_results, axis=0)
    print(f'Average Results: Test Acc: {avg_results[0]:.4f}, pre: {avg_results[1]:.4f}, rec: {avg_results[2]:.4f}, f1: {avg_results[3]:.4f}')


if __name__ == "__main__":
    main()
