#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import time
import argparse
from experimental_models import get_model
from preprocessing import load_data, create_graphs, convert_to_pyg_data, create_data_loaders
from helper import get_best_gpu, initialize_weights
from datetime import datetime

# Argument parsing
parser = argparse.ArgumentParser(description='Model Training Script')
parser.add_argument('--model', type=str, default='GCN', help='Model to train (GCN, GIN, MLP, CNN_1D, LSTM, RNN, MLP_Mixer, TF)')
parser.add_argument('--suppress_threshold', type=float, default=0.3, help='Threshold to suppress low connectivity')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--drop_out', type=float, default=0.6, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for leaky ReLU')
parser.add_argument('--layers', type=int, default=3, help='Number of layers in the model')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden features')
parser.add_argument('--out_features', type=int, default=8, help='Number of output features')
parser.add_argument('--n_classes', type=int, default=4, help='Number of classes')
parser.add_argument('--pth_dir', type=str, default='data/ppmi_w_curv.pth', help='Dataset Directory')
parser.add_argument('--suppress_ihc', action='store_true', help='Suppress all non-homotopic interhemispherical edges')
parser.add_argument('--use_asym', action='store_true', help='Uses Asymmetry index for homotopic edges')
parser.add_argument('--use_curvature', action='store_true', help='Uses curvature as additional features for edges')
parser.add_argument('--gpu', type=int, help='GPU index to use')
args = parser.parse_args()

# CONSTANTS
DATASET_PATH = args.pth_dir
NODE_FEATURES = 5
EDGE_FEATURES = 6

# HYPERPARAMETERS
SUPPRESS_THRESHOLD = args.suppress_threshold
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
FOLDS = args.folds
BATCH_SIZE = args.batch_size
DROP_OUT = args.drop_out
ALPHA = args.alpha
LAYERS = args.layers
HIDDEN = args.hidden
OUT_FEATURES = args.out_features
N_CLASSES = args.n_classes
SUPPRESS_IHC = args.suppress_ihc

# Print the arguments
print(f"Model: {args.model}")
print(f"Suppress Threshold: {args.suppress_threshold}")
print(f"Epochs: {args.epochs}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Folds: {args.folds}")
print(f"Batch Size: {args.batch_size}")
print(f"Dropout: {args.drop_out}")
print(f"Alpha: {args.alpha}")
print(f"Layers: {args.layers}")
print(f"Hidden Features: {args.hidden}")
print(f"Output Features: {args.out_features}")
print(f"Number of Classes: {args.n_classes}")
print(f"Suppress non-homotopic interhemispherical edges: {args.suppress_ihc}")
print(f"Use asymmetry index for homotopic edges: {args.use_asym}")
print(f"Use curvature as additional features for edges: {args.use_curvature}")

# Load and preprocess data
connectivity_matrices, connectivity_labels, curvatures, asymmetry_indices = load_data(
    dataset_path=DATASET_PATH,
    suppress_ihc=SUPPRESS_IHC
)

# Construct graphs
graphs = create_graphs(connectivity_matrices, curvatures, asymmetry_indices, SUPPRESS_THRESHOLD, args.use_curvature, args.use_asym)

# Convert to PyTorch Geometric data
graph_data_list = [convert_to_pyg_data(G) for G in graphs]
labels = torch.tensor(connectivity_labels, dtype=torch.long)

# Create data loaders
data_loaders = create_data_loaders(graph_data_list, labels, n_splits=FOLDS, batch_size=BATCH_SIZE)

# Check if CUDA is available
if torch.cuda.is_available():
    if args.gpu is None:
        best_gpu = get_best_gpu()
        device = torch.device(f'cuda:{best_gpu}')
        print(f'Using GPU {best_gpu} with the most free memory')
    else:
        device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Generate a unique log directory name using architecture and training parameters
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir_name = f'{args.model}_layers{args.layers}_hidden{args.hidden}_epochs{args.epochs}_lr{args.learning_rate}_bs{args.batch_size}_{timestamp}'
writer = SummaryWriter(log_dir=f'runs/{log_dir_name}')

def train_model(model, train_loader, optimizer, criterion, device, clip_value=2.0):
    model.train()
    total_loss = 0
    correct = 0
    for data in tqdm(train_loader, desc='Training', leave=False):
        data = data.to(device)
        
        # Check for NaNs and Infs in input data
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            raise ValueError("Input node features contain NaNs or Infs")
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            raise ValueError("Edge attributes contain NaNs or Infs")
        
        optimizer.zero_grad()
        out = model(data)
        
        if isinstance(out, tuple):
            out = out[0]  # Use only the first element if the model returns a tuple

        # Check if the outputs contain any NaNs or Infs
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("Model output contains NaNs or Infs")

        loss = criterion(out, data.y.view(-1))
        
        # Check if the loss is NaN or Inf
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError("Loss contains NaNs or Infs")
        
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        total_loss += loss.item() * data.num_graphs
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
            if isinstance(out, tuple):
                out = out[0]  # Use only the first element if the model returns a tuple
            loss = criterion(out, data.y.view(-1))
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += pred.eq(data.y.view(-1)).sum().item()
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return avg_loss, accuracy

for fold, (train_loader, test_loader) in enumerate(data_loaders):
    model = get_model(args, device, NODE_FEATURES, HIDDEN, OUT_FEATURES, LAYERS, N_CLASSES, connectivity_matrices)
    initialize_weights(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_wts = None

    for epoch in range(EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        elapsed_time = time.time() - start_time

        writer.add_scalar(f'Fold_{fold+1}/Train_Loss', train_loss, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Train_Acc', train_acc, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Test_Loss', test_loss, epoch)
        writer.add_scalar(f'Fold_{fold+1}/Test_Acc', test_acc, epoch)

        print(f'[Fold {fold + 1}][Epoch {epoch + 1}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {elapsed_time:.2f}s')

        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()

    model_save_path = f'models/best_models/{args.model}_{timestamp}_{fold}.pth'
    torch.save({
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }, model_save_path)
    print(f'Saved best model of fold {fold + 1} to {model_save_path}')

writer.close()
