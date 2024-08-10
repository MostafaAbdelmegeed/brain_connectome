import itertools
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN on connectivity data")
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset_name', type=str, default='ppmi', help='Name of the dataset')
    parser.add_argument('--seed', type=int, default=10, help='Random seed')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--filter_size', type=int, default=6, help='Filter size for PANConv')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    return parser.parse_args()

def run_experiment(args):
    command = [
        "python", "classifier/experimental3.py",
        "--gpu_id", str(args.gpu_id),
        "--dataset_name", args.dataset_name,
        "--seed", str(args.seed),
        "--n_folds", str(args.n_folds),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--hidden_dim", str(args.hidden_dim),
        "--dropout", str(args.dropout),
        "--filter_size", str(args.filter_size),
        "--heads", str(args.heads),
        "--patience", str(args.patience),
        "--test_size", str(args.test_size),
        "--pooling", args.pooling,
        "--percentile", str(args.percentile),
    ]
    subprocess.run(command)

def main():
    # Define the hyperparameter grid
    args = parse_args()
    dataset = args.dataset_name
    seed = args.seed

    hyperparameter_grid = {
        'gpu_id': [0],  # Assuming you want to stick to one GPU, or change this list if using multiple GPUs
        'dataset_name': [dataset],
        'seed': [seed],
        'n_folds': [10],
        'epochs': [300, 500],
        'batch_size': [1, 16],
        'learning_rate': [0.001, 0.0001],
        'hidden_dim': [16, 64, 256, 1024, 4096],
        'dropout': [0.5, 0.7],
        'filter_size': [6, 12, 18, 24],
        'heads': [1, 2, 3],
        'patience': [30, 50],
        'test_size': [0.1],
        'pooling': ['mean', 'max'],
        'percentile': [0.9, 0.95]
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*hyperparameter_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Run all experiments
    for experiment in experiments:
        args = parse_args()
        for key, value in experiment.items():
            setattr(args, key, value)
        print(f"Running experiment with args: {experiment}")
        run_experiment(args)

if __name__ == "__main__":
    main()