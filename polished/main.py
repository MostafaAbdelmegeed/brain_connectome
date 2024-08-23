from dataset import *
from models import *
from encoding import *
from train import *
import argparse
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on connectivity data")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset', type=str, default='ppmi', help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--filter_size', type=int, default=6, help='Filter size for PANConv')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    parser.add_argument('--augmented', action='store_true', help='Use augmented data')
    parser.add_argument('--model', type=str, default='BrainNet', help='Model name')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cpu'
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device('cuda:{}'.format(args.gpu_id))
        torch.set_default_device(device)
    torch.autograd.set_detect_anomaly(True)
    dataset_name = args.dataset
    data = torch.load(f'data/{dataset_name}_coembed_p{int(args.percentile*100)}{"_augmented" if args.augmented else ""}.pth')
    dataset = BrainDataset(data)
    train(args.model, dataset, device, args)

if __name__ == "__main__":
    main()