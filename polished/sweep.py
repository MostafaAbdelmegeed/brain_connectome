import subprocess
import itertools

# Define hyperparameters and their possible values
hyperparameters = {
    'seed': [0,1,2,3,4],
    'epochs': [300],
    'patience': [300],
    'model': ['gcn'],
    'edge_dim': [1],
    'n_folds': [10],
    'batch_size': [64],
    'gpu_id': [0],
    'dataset': ['adni'],
    'learning_rate': [0.0001],
    'dropout': [0.7],
    'hidden_dim': [1024],
    'heads': [2],
    'n_layers': [2],
    'in_channels': [58],
    'mgnn': [False],
    'num_classes': [2],
    'mode': ['asym']
}

# Generate all combinations of hyperparameters
keys = hyperparameters.keys()
combinations = list(itertools.product(*hyperparameters.values()))
# current_mod = 'fenc+edges-self_corr+cw+focal'
# current_mod = 'baseline'
current_mod = 'ASYM'

# Loop over each combination and run the training script
for idx, combo in enumerate(combinations):
    # Create a dictionary of the current hyperparameters
    params = dict(zip(keys, combo))
    run_name = f"BrainNet{params['model'].upper()}_s{params['seed']}_l{params['n_layers']}_h{params['hidden_dim']}_e{params['epochs']}_d{params['dropout']}_{current_mod}_{params['dataset']}"
    print(f"Running combination {idx + 1}/{len(combinations)} {current_mod}:\nRun name: {run_name}\n{params}")
    
    # Construct the command to run the training script
    cmd = [
        'python', '-u', 'polished/main.py',
        '--dataset', params['dataset'],
        '--model', params['model'],
        '--seed', str(params['seed']),
        '--gpu_id', str(params['gpu_id']),
        '--batch_size', str(params['batch_size']),
        '--n_folds', str(params['n_folds']),
        '--epochs', str(params['epochs']),
        '--patience', str(params['patience']),
        '--edge_dim', str(params['edge_dim']),
        '--n_layers', str(params['n_layers']),
        '--hidden_dim', str(params['hidden_dim']),
        '--dropout', str(params['dropout']),
        '--learning_rate', str(params['learning_rate']),
        '--heads', str(params['heads']),
        '--in_channels', str(params['in_channels']),
        '--num_classes', str(params['num_classes']),
        '--mode', params['mode'],
        '--run_name', run_name
    ]

    if params['mgnn']:
        cmd.append('--mgnn')
    
    log_file = f'polished/logs/{run_name}.log'
    with open(log_file, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)