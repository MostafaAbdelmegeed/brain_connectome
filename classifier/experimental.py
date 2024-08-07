import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear, Dropout, LayerNorm, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, global_add_pool, GATv2Conv, BatchNorm, PANConv
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from torch_geometric.data import Batch
import scipy.sparse as sp
import datetime
import argparse
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size for splitting data')
    parser.add_argument('--pooling', type=str, default='mean', help='Pooling method')
    parser.add_argument('--percentile', type=float, default=0.9, help='Percentile for thresholding')
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

class ConnectivityDataset(Dataset):
    def __init__(self, data):
        self.connectivities = data['connectivity']
        self.node_adjacencies = data['node_adj']
        self.edge_adjacencies = data['edge_adj']
        self.transition = data['transition']
        self.labels = data['label']
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
        # Ensure coalesced and print for debugging
        edge_adj = edge_adj.coalesce()
        transition = transition.coalesce()
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        for j in range(connectivity.shape[0]):
            for k in range(j + 1, connectivity.shape[0]):
                if node_adj[j, k] == 0:
                    continue
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isLeftIntra(j, k), self.isRightIntra(j, k), self.isHomo(j, k)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = label.type(torch.long)
        data = Data(x=torch.eye(connectivity.shape[0], dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=connectivity.shape[0], node_adj=node_adj, edge_adj=edge_adj, transition=transition)
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


data = torch.load(f'data/{dataset_name}_coembed_p{int(args.percentile*100)}.pth')

functional_groups = yeo_network()

class BrainEncoding(MessagePassing):
    def __init__(self, functional_groups, num_nodes, hidden_dim, atlas=116, dropout_rate=0.6, heads=1, edge_dim=5):
        super(BrainEncoding, self).__init__()
        self.functional_groups = functional_groups
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.atlas = atlas
        self.heads = heads
        
        # Linear layer to transform the one-hot encoding
        self.linear = Linear(self.num_nodes + self.hidden_dim + self.atlas // 2, self.hidden_dim)
        self.relu = LeakyReLU()
        self.dropout_rate = dropout_rate
        self.dropout = Dropout(dropout_rate)
        
        # Graph attention layer
        self.gat = GATv2Conv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=self.dropout_rate, edge_dim=edge_dim)
        
        self.encoding = self.create_encoding()

    @property
    def device(self):
        return next(self.parameters()).device

    def create_encoding(self):
        encoding = torch.zeros((self.num_nodes, self.hidden_dim + self.atlas // 2))
        for group, nodes in self.functional_groups.items():
            for node in nodes:
                region_encoding = [0] * (self.atlas // 2)
                region_encoding[node // 2] = 1 if node % 2 == 0 else -1
                encoding[node] = torch.tensor(region_encoding + self.get_group_encoding(group).tolist())
        return encoding

    def get_group_encoding(self, group):
        # Example encoding based on group name. You can customize this.
        encoding = torch.zeros(self.hidden_dim)
        encoding[hash(group) % self.hidden_dim] = 1
        return encoding

    def forward(self, data):
        if data.x is not None:
            batch_size = data.x.size(0) // self.num_nodes
            expanded_encoding = self.encoding.repeat(batch_size, 1)
            encodings = torch.cat([data.x, expanded_encoding], dim=-1)
        else:
            encodings = self.encoding

        # Apply linear transformation and ReLU activation
        encodings = self.relu(self.linear(encodings))
        encodings = self.dropout(encodings)
        
        # Apply graph attention layer
        encodings = self.gat(encodings, data.edge_index, data.edge_attr)
        
        data.x = encodings
        return data

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
        H_e = H_e.view(batch_size, -1, self.in_features_e)
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



class Net(torch.nn.Module):
    def __init__(self, functional_groups, num_nodes, hidden_dim, edge_dim, out_dim, heads=1, atlas=116, dropout=0.5, pooling='mean', filter_size=6):
        super(Net, self).__init__()
        self.brain_encoding = BrainEncoding(functional_groups=functional_groups, num_nodes=num_nodes, hidden_dim=hidden_dim, atlas=atlas)
        self.conv1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)
        self.conv2 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False)
        
        # self.gat = GATv2Conv(hidden_dim, hidden_dim//4, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.pan = PANConv(hidden_dim, hidden_dim//2, filter_size)
        self.lin = torch.nn.Linear(hidden_dim//2, out_dim)
        self.dropout = dropout
        self.pooling = pooling
        # Adding batch normalization layers
        self.bn_conv1 = BatchNorm(hidden_dim)
        self.bn_conv2 = BatchNorm(hidden_dim)

    def forward(self, data):
        #print_with_timestamp(f'data.x shape: {data.x.size()}, data.edge_attr shape: {data.edge_attr.size()}, data.edge_index shape: {data.edge_index.size()}, data.edge_adj shape: {data.edge_adj.size()}, data.node_adj shape: {data.node_adj.size()}, data.transition shape: {data.transition.size()}')
        data = self.brain_encoding(data)
        x, edge_attr = self.conv1(data.x, data.edge_attr, data.edge_adj, data.node_adj, data.transition)
        x = self.bn_conv1(x.reshape(x.shape[0]*x.shape[1], x.shape[2]))
        x, edge_attr = F.leaky_relu(x), F.leaky_relu(edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        x, edge_attr = self.conv2(x, edge_attr, data.edge_adj, data.node_adj, data.transition)
        x = self.bn_conv2(x.reshape(x.shape[0]*x.shape[1], x.shape[2]))
        x, edge_attr = F.leaky_relu(x), F.leaky_relu(edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)


        x, z = self.pan(x, data.edge_index)
        z = z.to_dense()
        x = x.to_dense()
        #print_with_timestamp(f'x shape: {x.size()}')

        if self.pooling == 'mean':
            x = global_mean_pool(x, data.batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, data.batch)
        elif self.pooling == 'add':
            x = global_add_pool(x, data.batch)
        x = self.lin(x)

        return x




    



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

test_size = args.test_size

n_nodes = 116
edge_dim = dataset.edge_features_count()
out_dim = 4 if dataset_name == 'ppmi' else 2
# Cross-validation setup
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
# Initialize lists to store metrics for all folds
all_fold_metrics = {'accuracy': [], 'precision': [], 'f1': []}

# TensorBoard writer
writer = SummaryWriter(log_dir=f'runs/{dataset_name}_experimental_s{seed}')
writer.add_text('Arguments', str(args))

# Training and evaluation loop
for fold, (train_index, test_index) in enumerate(kf.split(dataset, dataset.labels.to('cpu'))):
    print_with_timestamp(f"Fold {fold + 1}/{n_folds}")
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=test_size, random_state=seed)
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
    train_loader.collate_fn = collate_function
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=generator)
    val_loader.collate_fn = collate_function
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader.collate_fn = collate_function

    model = Net(functional_groups, n_nodes, hidden_dim, edge_dim, out_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for data in train_loader:
            #print(f'Batch size: {len(data)}')
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward(retain_graph=True)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        writer.add_scalar(f'Fold_{fold+1}/Train_Loss', train_loss, epoch)

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                val_loss += criterion(output, data.y).item()
        val_loss /= len(val_loader)
        writer.add_scalar(f'Fold_{fold+1}/Val_Loss', val_loss, epoch)
        print_with_timestamp(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}")

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
    preds = []
    labels = []
    for data in test_loader:
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted')
    writer.add_scalar(f'Fold_{fold+1}/Test_Accuracy', acc, epoch)
    writer.add_scalar(f'Fold_{fold+1}/Test_Precision', precision, epoch)
    writer.add_scalar(f'Fold_{fold+1}/Test_F1', f1, epoch)
    print_with_timestamp(f"Fold {fold + 1} Metrics: Accuracy: {acc}, Precision: {precision}, F1 Score: {f1}")

    # Store metrics for the current fold
    all_fold_metrics['accuracy'].append(acc)
    all_fold_metrics['precision'].append(precision)
    all_fold_metrics['f1'].append(f1)

    # Print average metrics until the latest fold
    avg_accuracy = np.mean(all_fold_metrics['accuracy'])
    avg_precision = np.mean(all_fold_metrics['precision'])
    avg_f1 = np.mean(all_fold_metrics['f1'])
    print_with_timestamp(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy}, Precision: {avg_precision}, F1 Score: {avg_f1}")

# Compute and print final metrics after cross-validation
final_accuracy = np.mean(all_fold_metrics['accuracy'])
final_precision = np.mean(all_fold_metrics['precision'])
final_f1 = np.mean(all_fold_metrics['f1'])

print_with_timestamp("Training completed.")
print_with_timestamp(f"Final Metrics:\nAccuracy: {final_accuracy}\nPrecision: {final_precision}\nF1 Score: {final_f1}")
# Close the TensorBoard writer
writer.close()