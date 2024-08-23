import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, Dropout, LayerNorm, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool, GATv2Conv, BatchNorm, PANConv, GCNConv, Linear, ResGatedGraphConv, PDNConv, AntiSymmetricConv, PANPooling, GINConv, Sequential
from torch_geometric.utils import dropout_node, dropout_edge, dropout_adj
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Batch
import scipy.sparse as sp
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter
import datetime

# python classifier/experimental6.py --gpu_id 0 --dataset_name ppmi --seed 10 --n_folds 10 --epochs 300 --batch_size 32 --learning_rate 0.00001 --hidden_dim 1024 --dropout 0.5 --heads 2 --patience 30 --test_size 0.2 --pooling mean --percentile 0.9 --n_layers 10


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")


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
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--filter_size', type=int, default=6, help='Filter size for PANConv')
    parser.add_argument('--heads', type=int, default=1, help='Number of attention heads')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
    parser.add_argument('--augmented', action='store_true', help='Use augmented data')
    return parser.parse_args()

args = parse_args()

print_with_timestamp(f"Arguments: {args}")

gpu_id = args.gpu_id

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

dataset_name = args.dataset_name


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.autograd.set_detect_anomaly(True)

def yeo_network():
    # Define the refined functional classes based on Yeo's 7 Network Parcellations
    visual = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    somatomotor = [0, 1, 57, 58, 69, 70, 16, 17, 18, 19]
    dorsal_attention = [59, 60, 61, 62, 67, 68, 6, 7]
    ventral_attention = [12, 13, 63, 64, 29, 30]
    limbic = [37, 38, 39, 40, 41, 42, 31, 32, 33, 34, 35, 36]
    frontoparietal = [2, 3, 4, 5, 6, 7, 10, 11]
    default_mode = [23, 24, 67, 68, 65, 66, 35, 36]
    functional_groups = {}
    functional_groups['visual'] = visual
    functional_groups['somatomotor'] = somatomotor
    functional_groups['dorsal_attention'] = dorsal_attention
    functional_groups['ventral_attention'] = ventral_attention
    functional_groups['limbic'] = limbic
    functional_groups['frontoparietal'] = frontoparietal
    functional_groups['default_mode'] = default_mode
    functional_groups['other'] = []
    for i in range(116):
        if i not in visual + somatomotor + dorsal_attention + ventral_attention + limbic + frontoparietal + default_mode:
            functional_groups['other'].append(i)
    return functional_groups

def is_undirected(edge_index):
    # Convert edge_index to a set of tuples (i, j) for easier comparison
    edge_set = set(map(tuple, edge_index.t().tolist()))

    # Check if for every edge (i, j) the reverse edge (j, i) exists
    for i, j in edge_set:
        if (j, i) not in edge_set:
            return False
    
    return True


class ConnectivityDataset(Dataset):
    def __init__(self, data):
        self.connectivities = data['connectivity']
        self.node_adjacencies = data['node_adj']
        self.edge_adjacencies = data['edge_adj']
        self.transition = data['transition']
        self.labels = data['label'].type(torch.long)
        self.node_num = self.connectivities[0].shape[0]
        self.left_indices, self.right_indices = self.get_hemisphere_indices()
    
    def get_hemisphere_indices(self):
        left_indices = [i for i in range(self.node_num) if i % 2 == 0]
        right_indices = [i for i in range(self.node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self, i, j):
        return (i in self.left_indices and j in self.right_indices) or (i in self.right_indices and j in self.left_indices)

    def isLeftIntra(self, i, j):
        return i in self.left_indices and j in self.left_indices

    def isRightIntra(self, i, j):
        return i in self.right_indices and j in self.right_indices

    def isHomo(self, i, j):
        return i // 2 == j // 2 and abs(i - j) == 1
    
    def edge_features_count(self):
        return 5
    
    def node_features_count(self):
        return 116

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        node_adj = self.node_adjacencies[idx]
        edge_adj = self.edge_adjacencies[idx]
        transition = self.transition[idx]

        # Ensure coalesced
        edge_adj = edge_adj.coalesce()
        transition = transition.coalesce()
        label = self.labels[idx]

        # Initialize lists for edge_index and edge_attr
        edge_index = []
        edge_attr = []

        # Populate edge_index and edge_attr with additional features
        for j in range(connectivity.shape[0]):
            for k in range(connectivity.shape[0]):
                if node_adj[j, k] == 0:
                    continue
                edge_index.append([j, k])
                # Include additional edge attributes like isInter, isLeftIntra, etc.
                edge_attr.append([
                    connectivity[j, k], 
                    self.isInter(j, k), 
                    self.isLeftIntra(j, k), 
                    self.isRightIntra(j, k), 
                    self.isHomo(j, k)
                ])
        
        # Convert lists to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        # print_with_timestamp(f'edge_index: {edge_index.size()}, edge_attr: {edge_attr.size()}')
        # Create the Data object
        data = Data(
            x=torch.eye(connectivity.shape[0], dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label.type(torch.long),
            num_nodes=connectivity.shape[0],
            node_adj=node_adj,
            edge_adj=edge_adj,
            transition=transition
        )
        
        return data


def pad_edge_index(edge_index, max_edges):
    pad_size = max_edges - edge_index.size(1)
    if pad_size > 0:
        pad = torch.zeros((2, pad_size), dtype=edge_index.dtype, device=edge_index.device)
        edge_index = torch.cat([edge_index, pad], dim=1)
    return edge_index

def pad_edge_attr(edge_attr, max_edges, feature_dim):
    pad_size = max_edges - edge_attr.size(0)
    if pad_size > 0:
        pad = torch.zeros((pad_size, feature_dim), dtype=edge_attr.dtype, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, pad], dim=0)
    return edge_attr

def pad_matrix(matrix, max_nodes):
    pad_size = max_nodes - matrix.size(0)
    if pad_size > 0:
        pad = torch.zeros((pad_size, matrix.size(1)), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad], dim=0)
        pad = torch.zeros((matrix.size(0), pad_size), dtype=matrix.dtype, device=matrix.device)
        matrix = torch.cat([matrix, pad], dim=1)
    return matrix

def collate_function(batch):
    max_nodes = max([data.num_nodes for data in batch])
    max_edges = max([data.edge_index.size(1) for data in batch])
    feature_dim = batch[0].edge_attr.size(1)
    batched_data = []
    for data in batch:
        x = torch.eye(max_nodes, dtype=data.x.dtype, device=data.x.device)
        x[:data.num_nodes, :data.num_nodes] = data.x
        
        edge_index = pad_edge_index(data.edge_index, max_edges)
        edge_attr = pad_edge_attr(data.edge_attr, max_edges, feature_dim)
        
        node_adj = pad_matrix(data.node_adj, max_nodes)
        edge_adj = pad_matrix(data.edge_adj.to_dense(), max_edges)
        transition = pad_matrix(data.transition.to_dense(), max_nodes)  
        batched_data.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=data.y, num_nodes=max_nodes,
                                 node_adj=node_adj, edge_adj=edge_adj, transition=transition.to_sparse()))
    return Batch.from_data_list(batched_data)


data = torch.load(f'data/{dataset_name}_coembed_p{int(args.percentile*100)}{"_augmented" if args.augmented else ""}.pth')

functional_groups = yeo_network()

class BrainEncodeEmbed(MessagePassing):
    def __init__(self, functional_groups, hidden_dim, edge_dim, n_roi=116):
        super(BrainEncodeEmbed, self).__init__()
        self.functional_groups = functional_groups
        self.n_groups = len(self.functional_groups)
        self.hidden_dim = hidden_dim
        self.num_nodes = n_roi
        self.encoding = self.create_encoding()
        self.linear = Linear(n_roi+self.n_groups, hidden_dim)
        self.coembed_1 = AlternateConvolution(in_features_v=self.hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False)
        self.coembed_2 = AlternateConvolution(in_features_v=self.hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def create_encoding(self):
        encoding = torch.zeros((self.num_nodes, self.n_groups))
        for group, nodes in self.functional_groups.items():
            for node in nodes:
                group_encoding = torch.tensor(self.get_group_encoding(group).tolist())
                if node % 2 == 0:
                    # Left hemisphere
                    encoding[node] += group_encoding
                else:
                    # Right hemisphere
                    encoding[node] += group_encoding + 1  # Shift the encoding for the right hemisphere
        return encoding


    def get_group_encoding(self, group):
        # Example encoding based on group name. You can customize this.
        encoding = torch.zeros(self.n_groups)
        encoding[hash(group) % self.n_groups] = 1
        return encoding

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        n_adj = data.node_adj
        e_adj = data.edge_adj
        transition = data.transition
        batch_size = data.num_graphs
        expanded_encoding = self.encoding.repeat(batch_size, 1)
        if x is not None:
            x = torch.cat([x, expanded_encoding], dim=-1)
        x = self.linear(x)
        x, edge_attr = self.coembed_1(x, edge_attr, e_adj, n_adj, transition)
        x = x.reshape(data.num_nodes, -1, x.size(2)).squeeze(1)
        edge_attr = edge_attr.reshape(data.num_edges, -1, edge_attr.size(2)).squeeze(1)
        x, edge_attr = self.coembed_2(x, edge_attr, e_adj, n_adj, transition)
        x = x.reshape(data.num_nodes, -1, x.size(2)).squeeze(1)
        edge_attr = edge_attr.reshape(data.num_edges, -1, edge_attr.size(2)).squeeze(1)
        return x, edge_attr

class AlternateConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(AlternateConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float())
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        device = H_v.device  # Ensure all operations are on the same device
        batch_size = adj_v.shape[0]//adj_v.shape[1]
        # print_with_timestamp(f'H_v: {H_v.size()}, H_e: {H_e.size()}, adj_e: {adj_e.size()}, adj_v: {adj_v.size()}, T: {T.size()}')
        H_v = H_v.view(batch_size, -1, self.in_features_v)
        H_e = H_e.view(batch_size, -1, self.in_features_e).type(torch.float32)
        adj_e = adj_e.view(batch_size, -1, adj_e.size(1), adj_e.size(1)).squeeze(1)
        adj_v = adj_v.view(batch_size, -1, adj_v.size(1), adj_v.size(1)).squeeze(1)
        T = T.to_dense().view(batch_size, -1, T.size(0)//batch_size, T.size(1)).squeeze(1)
        # print_with_timestamp(f'H_v: {H_v.size()}, H_e: {H_e.size()}, adj_e: {adj_e.size()}, adj_v: {adj_v.size()}, T: {T.size()}')
        # print_with_timestamp(f'p: {self.p.size()}')
        if self.node_layer:
            H_e_p = H_e @ self.p.t()  # Adapted for batch
            diag_values = H_e_p.transpose(1, 2)
            diag_matrix = torch.diag_embed(diag_values).transpose(1, 2).squeeze(2)
            # print_with_timestamp(f'diag_matrix: {diag_matrix.size()}')
            # print_with_timestamp(f'T: {T.size()}')
            multiplier1 = torch.bmm(T, diag_matrix) @ T.transpose(1, 2)
            mask1 = torch.eye(multiplier1.size(1), device=device).unsqueeze(0).repeat(multiplier1.size(0), 1, 1)
            M1 = mask1 * torch.ones_like(multiplier1) + (1. - mask1) * multiplier1
            adjusted_A = torch.mul(M1, adj_v)
            weight_repeated = self.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            output = torch.bmm(adjusted_A, torch.bmm(H_v, weight_repeated))
            if self.bias is not None:
                output += self.bias
            return output, H_e
        else:
            H_v_p = H_v @ self.p.t()
            diag_values = H_v_p.transpose(1, 2)
            diag_matrix = torch.diag_embed(diag_values).squeeze(1)
            # print_with_timestamp(f'diag_matrix: {diag_matrix.size()}')
            # print_with_timestamp(f'T: {T.size()}')
            multiplier2 = torch.bmm(T.transpose(1, 2), diag_matrix) @ T
            mask2 = torch.eye(multiplier2.size(1), device=device).unsqueeze(0).repeat(multiplier2.size(0), 1, 1)
            M3 = mask2 * torch.ones_like(multiplier2) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e)
            normalized_adjusted_A = adjusted_A / (adjusted_A.max(2, keepdim=True)[0] + 1e-10)
            weight_repeated = self.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            output = torch.bmm(normalized_adjusted_A, torch.bmm(H_e, weight_repeated))
            if self.bias is not None:
                output += self.bias
            return H_v, output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features_v) + ' -> ' + str(self.out_features_v) + '), (' + str(self.in_features_e) + ' -> ' + str(self.out_features_e) + ')'


class BrainBlock(Module):
    def __init__(self, in_features, out_features, dropout=0.7):
        super(BrainBlock, self).__init__()
        self.gcn = GCNConv(in_features, out_features, add_self_loops=True, normalize=True)
        self.bn = BatchNorm(out_features)
        self.relu = LeakyReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.relu(x)
        x = self.bn(x)
        x= self.dropout(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, functional_groups, hidden_dim, edge_dim, out_dim, heads=1, atlas=116, dropout=0.7, pooling='mean', n_layers=1):
        super(Net, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_dim, edge_dim=edge_dim, n_roi=atlas)
        self.bn = BatchNorm(hidden_dim)
        self.layers = torch.nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(BrainBlock(hidden_dim, hidden_dim, dropout=dropout))
        self.fc = Linear(hidden_dim, out_dim)
        self.pooling = pooling

    def forward(self, data):
        # print_with_timestamp(f'data.x shape: {data.x.size()}, data.edge_attr shape: {data.edge_attr.size()}, data.edge_index shape: {data.edge_index.size()}, data.edge_adj shape: {data.edge_adj.size()}, data.node_adj shape: {data.node_adj.size()}, data.transition shape: {data.transition.size()}')
        # print_with_timestamp(f'Layer {0.0}| x mean: {data.x.mean()}, x std: {data.x.std()}')
        x, edge_attr = self.encemb(data)
        # print_with_timestamp(f'Layer {1.0}| x mean: {data.x.mean()}, x std: {data.x.std()}')
        for i, layer in enumerate(self.layers):
            x = layer(x, data.edge_index)
            # print_with_timestamp(f'Layer {i+2}.{i%5}| x mean: {x.mean()}, x std: {x.std()}')
        

        if self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, data.batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, data.batch)
        x = self.fc(x)


        return x
    

# class FocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, reduction='mean', device='cpu'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction

#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha]).to(device)
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha).to(device)

#     def forward(self, input, target):
#         # Convert logits to probabilities
#         p_t = F.softmax(input, dim=1)
#         target = target.view(-1, 1)
#         p_t = p_t.gather(1, target)
#         p_t = p_t.view(-1)

#         # Compute the focal loss
#         loss = - (1 - p_t) ** self.gamma * torch.log(p_t)

#         # Apply alpha weighting
#         if self.alpha is not None:
#             alpha_t = self.alpha.gather(0, target.view(-1))
#             loss = loss * alpha_t

#         if self.reduction == 'mean':
#             return loss.mean().to(self.alpha.device)
#         elif self.reduction == 'sum':
#             return loss.sum().to(self.alpha.device)
#         else:
#             return loss




    



dataset = ConnectivityDataset(data)
generator = torch.Generator(device=device)





def check_for_nans(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print_with_timestamp(f"NaN detected in {tensor_name}")



# Hyperparameters
n_folds = args.n_folds
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.learning_rate
hidden_dim = args.hidden_dim
dropout = args.dropout
patience = args.patience
pooling = args.pooling
heads = args.heads
filter_size = args.filter_size
n_layers = args.n_layers

test_size = args.test_size

n_nodes = 116
edge_dim = dataset.edge_features_count()
out_dim = 4 if dataset_name == 'ppmi' else 2
# Cross-validation setup
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
# Initialize lists to store metrics for all folds
all_fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'conf_matrix':[]}


# Get the current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# TensorBoard writer
writer = SummaryWriter(log_dir=f'runs/experimental_4_{dataset_name}_s{seed}_{timestamp}')
writer.add_text('Arguments', str(args))

 # Initialize gradient accumulators
epoch_gradients = {}

# Training and evaluation loop
for fold, (train_index, test_index) in enumerate(kf.split(dataset, dataset.labels.to('cpu'))):
    print_with_timestamp(f"Fold {fold + 1}/{n_folds}")
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)
    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=test_size, random_state=seed)
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    # Print label distributions
    train_labels = [dataset.labels[idx].item() for idx in train_index]
    val_labels = [dataset.labels[train_index[idx]].item() for idx in val_indices]
    test_labels = [dataset.labels[idx].item() for idx in test_index]
    
    print_with_timestamp(f"Training labels distribution: {np.bincount(train_labels)}")
    print_with_timestamp(f"Validation labels distribution: {np.bincount(val_labels)}")
    print_with_timestamp(f"Test labels distribution: {np.bincount(test_labels)}")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
    train_loader.collate_fn = collate_function
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=generator)
    val_loader.collate_fn = collate_function
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader.collate_fn = collate_function

    model = Net(functional_groups, hidden_dim, edge_dim, out_dim, dropout=dropout, heads=heads, n_layers=n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5, lr=learning_rate)
    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience//2)
     # Initialize gradients dictionary for this fold
    fold_gradients = {}

    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(dataset.labels.to('cpu').numpy()), y=dataset.labels.to('cpu').numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print_with_timestamp(f'Class weights: {class_weights}')
    print_with_timestamp(f"Epoch/Loss\t||\tTraining\t|\tValidation\t")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)

    best_val_loss = float('inf')
    patience_counter = 0

   

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0
        all_preds = []
        all_labels = []

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

            # Calculate predictions and accuracy
            preds = output.argmax(dim=1)
            correct_predictions += (preds == data.y).sum().item()
            total_predictions += data.y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

            # Store gradients for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in fold_gradients:
                        fold_gradients[name] = []
                    fold_gradients[name].append(param.grad.clone().detach().cpu())

        train_accuracy = correct_predictions / total_predictions
        train_loss /= len(train_loader)

        # Log metrics to TensorBoard
        writer.add_scalar(f'Fold_{fold+1}/Metrics/Train_Loss', train_loss, epoch)

        # Confusion matrix for the training set
        train_conf_matrix = confusion_matrix(all_labels, all_preds)
        # print_with_timestamp(f"Training Confusion Matrix (Epoch {epoch + 1}):\n{train_conf_matrix}")

        # print_with_timestamp(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
         # After the epoch, save the mean and std of the gradients
        for name, grads in fold_gradients.items():
            grads_tensor = torch.stack(grads)
            mean_grad = grads_tensor.mean(dim=0)
            std_grad = grads_tensor.std(dim=0)

            writer.add_scalar(f'Fold_{fold+1}/Gradients/{name}_mean', mean_grad.mean().item(), epoch)
            if std_grad.numel() > 1:
                writer.add_scalar(f'Fold_{fold+1}/Gradients/{name}_std', std_grad.mean().item(), epoch)

        # Clear gradients storage after saving to TensorBoard to save memory
        fold_gradients = {}

        # Validate the model
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data.y)
                val_loss += loss.item()
                
                preds = output.argmax(dim=1)
                val_correct += (preds == data.y).sum().item()
                val_total += data.y.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())

        val_accuracy = val_correct / val_total
        val_loss /= len(val_loader)
        val_conf_matrix = confusion_matrix(val_labels, val_preds)
        # print_with_timestamp(f"Validation Confusion Matrix (Epoch {epoch + 1}):\n{val_conf_matrix}")

        writer.add_scalar(f'Fold_{fold+1}/Metrics/Val_Loss', val_loss, epoch)

        for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f'Fold_{fold+1}/Gradients/{name}_mean', param.grad.mean(), epoch)
                    
                    # Only log std if the number of elements is greater than 1
                    if param.grad.numel() > 1:
                        writer.add_scalar(f'Fold_{fold+1}/Gradients/{name}_std', param.grad.std(), epoch)
        
        print_with_timestamp(f"{epoch + 1}/{epochs}\t\t||\t{train_loss:.4f}\t\t|\t{val_loss:.4f}\t")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print_with_timestamp(f"Early stopping at epoch {epoch + 1}")
                break

    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Evaluate the model on the test set
    model.eval()
    test_preds = []
    test_labels = []

    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            preds = output.argmax(dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(data.y.cpu().numpy())



    # Compute metrics
    acc = accuracy_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, test_preds, average='weighted')
    
    # Confusion matrix for the test set
    conf_matrix = confusion_matrix(test_labels, test_preds)
    # print_with_timestamp(f'Test Confusion Matrix:\n{conf_matrix}')

    writer.add_scalar(f'Fold_{fold+1}/Test_Accuracy', acc, epoch)
    writer.add_scalar(f'Fold_{fold+1}/Test_Precision', precision, epoch)
    writer.add_scalar(f'Fold_{fold+1}/Test_Recall', recall, epoch)
    writer.add_scalar(f'Fold_{fold+1}/Test_F1', f1, epoch)
    print_with_timestamp(f"Fold {fold + 1} Metrics: Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Store metrics for the current fold
    all_fold_metrics['accuracy'].append(acc)
    all_fold_metrics['precision'].append(precision)
    all_fold_metrics['recall'].append(recall)
    all_fold_metrics['f1'].append(f1)
    all_fold_metrics['conf_matrix'].append(conf_matrix)

    # Print average metrics until the latest fold
    avg_accuracy = np.mean(all_fold_metrics['accuracy'])
    avg_precision = np.mean(all_fold_metrics['precision'])
    avg_recall = np.mean(all_fold_metrics['recall'])
    avg_f1 = np.mean(all_fold_metrics['f1'])
    print_with_timestamp(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")



# Compute and print final metrics after cross-validation
final_accuracy = np.mean(all_fold_metrics['accuracy'])
final_precision = np.mean(all_fold_metrics['precision'])
final_recall = np.mean(all_fold_metrics['recall'])
final_f1 = np.mean(all_fold_metrics['f1'])

print_with_timestamp("Training completed.")
print_with_timestamp(f"Final Metrics | Accuracy: {final_accuracy:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f} | F1 Score: {final_f1:.4f}")

# Print the average confusion matrix
avg_conf_matrix = np.mean(all_fold_metrics['conf_matrix'], axis=0)
print_with_timestamp(f"Final Confusion Matrix:\n{avg_conf_matrix}")

# Close the TensorBoard writer
writer.close()