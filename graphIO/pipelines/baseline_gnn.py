import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from nilearn import datasets, plotting
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
from multi_head_gat_with_edge_features import GATWithEdgeFeatures
import torch.nn as nn
from tqdm import tqdm
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

def center_matrices(matrices):
    return standardize_matrices(normalize_matrices(matrices))
def normalize_matrices(matrices):
    min_val = np.min(matrices)
    max_val = np.max(matrices)
    normalized = (matrices - min_val) / (max_val - min_val)
    return normalized
def standardize_matrices(matrices):
    mean_val = np.mean(matrices)
    std_dev = np.std(matrices)
    standardized = (matrices - mean_val) / std_dev
    return standardized


# Argument parsing
parser = argparse.ArgumentParser(description='Brain Connectivity Analysis with GNNs')
parser.add_argument('--suppress_threshold', type=float, default=0.3, help='Threshold to suppress low connectivity')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
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
parser.add_argument('--center_matrices', action='store_true', help='Center matrices')
parser.add_argument('--normalize_matrices', action='store_true', help='Normalize matrices')
parser.add_argument('--standardize_matrices', action='store_true', help='Standardize matrices')
parser.add_argument('--model', type=str, default='GCN', help='Model to use')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
args = parser.parse_args()


# Print the arguments
print(f"Suppress Threshold: {args.suppress_threshold}")
print(f"Attention Heads: {args.attention_heads}")
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

# CONSTANTS
DATASET_PATH = args.pth_dir
STUDY_CONNECTOME_INDEX = 20
NODE_FEATURES = 5
EDGE_FEATURES = 6

# HYPERPARAMETERS
SUPPRESS_THRESHOLD = args.suppress_threshold
ATTENTION_HEADS = args.attention_heads
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

ppmi_dataset = torch.load(DATASET_PATH)
connectivity_matrices = ppmi_dataset['data'].numpy()
connectivity_labels = ppmi_dataset['class_label'].numpy().reshape(-1, 1)
curvatures = ppmi_dataset['curvatures'].numpy()
asymmetry_indices = np.array([
        calculate_inter_hemispheric_asymmetry_vector_aal116(matrix) 
        for matrix in tqdm(connectivity_matrices, desc='Computing asymmetry indices')
    ])
if args.center_matrices:
    connectivity_matrices = center_matrices(connectivity_matrices)
    curvatures = center_matrices(curvatures)
    asymmetry_indices = center_matrices(asymmetry_indices)
    print(f'Centered Data')
elif args.normalize_matrices:
    connectivity_matrices = normalize_matrices(connectivity_matrices)
    curvatures = normalize_matrices(curvatures)
    asymmetry_indices = normalize_matrices(asymmetry_indices)
    print(f'Normalized Data')
elif args.standardize_matrices:
    connectivity_matrices = standardize_matrices(connectivity_matrices)
    curvatures = standardize_matrices(curvatures)
    asymmetry_indices = standardize_matrices(asymmetry_indices)
    print(f'Standardized Data')