from dataset import *
from models import *
from networks import *
from train import *
import argparse
import torch
import numpy as np
import random

# python -u main.py --dataset ppmi --seed 0 --n_folds 10 --epochs 300 --patience 50 --batch_size 64 --learning_rate 0.0001 --hidden_dim 1024 --n_layers 2 --dropout 0.7 --heads 1 --test_size 0.1 --percentile 0.9 --model gin --gpu_id 0
def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on connectivity data")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset', type=str, default='ppmi', help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    parser.add_argument('--model', type=str, default='gin', help='Model name')
    parser.add_argument('--vanilla', action='store_true', help='Use vanilla dataset')
    parser.add_argument('--exp_code', type=str, default='-', help='Experiment code')
    parser.add_argument('--run_name', type=str, default='', help='Run name')
    parser.add_argument('--edge_dim', type=int, default=1, help='Edge feature dimension')
    parser.add_argument('--in_channels', type=int, default=1, help='Input channel size')
    parser.add_argument('--mgnn', action='store_true', help='Use MGNN')
    parser.add_argument('--num_classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--mode', type=str, default='vanilla', help='Mode', choices=['vanilla', 'function', 'both'])
    parser.add_argument('--network', type=str, default='yeo7', help='Network', choices=['yeo17', 'yeo7'])
    parser.add_argument('--include_asymmetry', action='store_true', help='Include asymmetry')
    parser.add_argument('--use_edges', action='store_true', help='Use edge features')
    parser.add_argument('--include_connection_type', action='store_true', help='Use node features')
    return parser.parse_args()


def main():
    args = parse_args()
    print_with_timestamp(args)
    device = 'cpu'
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_printoptions(threshold=torch.inf)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        device = torch.device('cuda:{}'.format(args.gpu_id))
        # torch.set_default_device(device)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train(args, device)

if __name__ == "__main__":
    main()