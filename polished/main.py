from dataset import *
from models import *
from encoding import *
from train import *
import argparse
import torch
import numpy as np

# python polished/main.py --gpu_id 0 --dataset ppmi --seed 10 --n_folds 10 --epochs 300 --batch_size 8 --learning_rate 0.0001 --hidden_dim 256 --n_layers 1 --dropout 0.5 --heads 1 --patience 30 --test_size 0.2 --percentile 0.9 --augmented --model brain

def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

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
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    parser.add_argument('--augmented', action='store_true', help='Use augmented data')
    parser.add_argument('--augment_validation', action='store_true', help='Augment validation data')
    parser.add_argument('--span', type=float, default=0.02, help='Span for augmented data')
    parser.add_argument('--model', type=str, default='brain', help='Model name')
    parser.add_argument('--vanilla', action='store_true', help='Use vanilla dataset')
    parser.add_argument('--exp_code', type=str, default='-', help='Experiment code')
    parser.add_argument('--run_name', type=str, default='', help='Run name')
    return parser.parse_args()


def main():
    args = parse_args()
    print_with_timestamp(args)
    device = 'cpu'
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device('cuda:{}'.format(args.gpu_id))
        # torch.set_default_device(device)
    torch.autograd.set_detect_anomaly(True)
    train(args, device)

if __name__ == "__main__":
    main()