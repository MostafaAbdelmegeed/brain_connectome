import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, Linear
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
import math
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from torch_geometric.data import Batch
import scipy.sparse as sp
import datetime


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

gpu_id = 0

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

dataset_name = 'ppmi'


seed = 10
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
    def __init__(self, connectivities, labels):
        self.connectivities = connectivities
        self.node_num = len(connectivities[0])
        self.labels = labels
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

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        for j in range(connectivity.shape[0]):
            for k in range(j + 1, connectivity.shape[0]):
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isLeftIntra(j, k), self.isRightIntra(j, k), self.isHomo(j, k)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = label.type(torch.long)
        data = Data(x=None, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=connectivity.shape[0])
        return data

    


data = torch.load(f'data/{dataset_name}.pth')
connectivities = data['matrix']
print_with_timestamp(f'Connectivities mean: {connectivities.mean()}, std: {connectivities.std()}, min: {connectivities.min()}, max: {connectivities.max()}')


connectivities = (connectivities - connectivities.mean()) / (connectivities.std()+ 1e-8)
labels = data['label']
functional_groups = yeo_network()

class BrainEncoding(MessagePassing):
    def __init__(self, functional_groups, num_nodes, hidden_dim, atlas=116):
        super(BrainEncoding, self).__init__()
        self.functional_groups = functional_groups
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.atlas = atlas
        # Linear layer to transform the one-hot encoding
        self.linear = Linear(self.hidden_dim + self.atlas // 2, self.hidden_dim)
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
        encoding = self.linear(encoding)
        return encoding

    def get_group_encoding(self, group):
        # Example encoding based on group name. You can customize this.
        encoding = torch.zeros(self.hidden_dim)
        encoding[hash(group) % self.hidden_dim] = 1
        return encoding

    def forward(self, batch):
        batch_size = batch.num_graphs
        encodings = self.encoding.repeat(batch_size, 1, 1).squeeze(0)
        if batch.x is None:
            batch.x = encodings
        else:
            batch.x = torch.cat([batch.x, encodings], dim=-1)
        return batch

class AlternateConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True):
        super(AlternateConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v

        if node_layer:
            self.node_layer = True
            self.weight = Parameter(torch.FloatTensor(in_features_v, out_features_v)).to(device)
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_e))).float()).to(device)
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_v)).to(device)
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = Parameter(torch.FloatTensor(in_features_e, out_features_e)).to(device)
            self.p = Parameter(torch.from_numpy(np.random.normal(size=(1, in_features_v))).float()).to(device)
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features_e)).to(device)
            else:
                self.register_parameter('bias', None)
        self.reset_parameters()
    
    @property
    def device(self):
        return next(self.parameters()).device

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
            mask1 = torch.eye(multiplier1.shape[0])
            M1 = mask1 * torch.ones(multiplier1.shape[0]) + (1. - mask1)*multiplier1
            adjusted_A = torch.mul(M1, adj_v.to_dense())
            '''
            print_with_timestamp("adjusted_A is ", adjusted_A)
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            print_with_timestamp("normalized adjusted A is ", normalized_adjusted_A)
            '''
            # to avoid missing feature's influence, we don't normalize the A
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e
        else:
            diag_values = (H_v @ self.p.t()).t()[0]
            multiplier2 = torch.spmm(T.t(), torch.diag(diag_values)) @ T.to_dense()
            mask2 = torch.eye(multiplier2.shape[0], device=multiplier2.device)
            M3 = mask2 * torch.ones(multiplier2.shape[0], device=multiplier2.device) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e.to_dense())
            max_values = adjusted_A.max(0, keepdim=True)[0]
            normalized_adjusted_A = adjusted_A / (max_values + 1e-10)  # Add epsilon to avoid division by zero
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features_v) + ' -> ' + str(self.out_features_v) + '), (' + str(self.in_features_e) + ' -> ' + str(self.out_features_e) + ')'


def sparse_mx_to_torch_sparse_tensor(sparse_mx, device=None):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.tensor([sparse_mx.row, sparse_mx.col], dtype=torch.int64, device=device)
    values = torch.tensor(sparse_mx.data, dtype=torch.float32, device=device)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def node_adjacency_matrix(edge_index, num_nodes):
    """Create a node adjacency matrix from an edge index."""
    # Initialize an adjacency matrix with zeros on the GPU
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=edge_index.device)
    # Iterate over the edges and set the corresponding entries in the adjacency matrix to 1
    for i in range(edge_index.shape[1]):
        node1 = edge_index[0, i]
        node2 = edge_index[1, i]
        adj_matrix[node1, node2] = 1
        adj_matrix[node2, node1] = 1  # If the graph is undirected

    # Ensure the diagonal is zero
    adj_matrix.fill_diagonal_(0)
    
    return adj_matrix
    # # Initialize an adjacency matrix with zeros
    # adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
    
    # # Set the corresponding entries in the adjacency matrix to 1
    # adj_matrix[edge_index[0], edge_index[1]] = 1
    # adj_matrix[edge_index[1], edge_index[0]] = 1  # If the graph is undirected
    
    # return adj_matrix

    
def edge_adjacency_matrix(node_adj):
    """Create an edge adjacency matrix from a node adjacency matrix."""
    node_adj.fill_diagonal_(0)
    num_nodes = node_adj.shape[0]
    num_edges = num_nodes * (num_nodes - 1) // 2
    edge_adj = torch.ones((num_edges, num_edges), dtype=torch.float, device=node_adj.device) - torch.eye(num_edges, device=node_adj.device)
    return edge_adj
    # # Ensure node_adj is on CPU and convert to numpy
    # node_adj = node_adj.cpu().numpy()
    
    # # Fill the diagonal with zeros
    # np.fill_diagonal(node_adj, 0)
    
    # # Get the upper triangular part of the adjacency matrix
    # edge_index = np.triu(node_adj).nonzero()
    
    # # Number of edges
    # num_edge = len(edge_index[0])
    
    # # Initialize the edge adjacency matrix with zeros
    # edge_adj = np.zeros((num_edge, num_edge))
    
    # # Fill the edge adjacency matrix
    # for i in range(num_edge):
    #     for j in range(i, num_edge):
    #         if np.intersect1d(edge_index[0][i], edge_index[1][j]).size > 0:
    #             edge_adj[i, j] = 1
    
    # # Make the matrix symmetric and fill the diagonal with ones
    # edge_adj = edge_adj + edge_adj.T
    # np.fill_diagonal(edge_adj, 1)
    
    # # Convert to sparse matrix and then to torch sparse tensor
    # return sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(edge_adj))



def transition_matrix(node_adj):
    """Create a transition matrix from a node adjacency matrix."""
    node_adj.fill_diagonal_(0)
    edge_index = torch.nonzero(torch.triu(node_adj)).t()
    num_edges = edge_index.shape[1]
    row_index = edge_index.t().flatten()
    col_index = torch.repeat_interleave(torch.arange(num_edges, device=node_adj.device), 2)
    data = torch.ones(num_edges * 2, device=node_adj.device)
    T = torch.sparse_coo_tensor(torch.stack([row_index, col_index]), data, (node_adj.shape[0], num_edges))
    return T
    # # Ensure node_adj is on CPU and convert to numpy
    # node_adj = node_adj.cpu().numpy()
    
    # # Fill the diagonal with zeros
    # np.fill_diagonal(node_adj, 0)
    
    # # Get the upper triangular part of the adjacency matrix
    # edge_index = np.triu(node_adj).nonzero()
    
    # # Number of edges
    # num_edge = len(edge_index[0])
    
    # # Create row and column indices for the transition matrix
    # row_index = np.repeat(np.arange(num_edge), 2)
    # col_index = np.hstack([edge_index[0], edge_index[1]])
    
    # # Data values for the transition matrix
    # data = np.ones(len(row_index))
    
    # # Create the transition matrix as a sparse matrix
    # T = sp.csr_matrix((data, (row_index, col_index)), shape=(num_edge, node_adj.shape[0]))
    
    # return sparse_mx_to_torch_sparse_tensor(T)



class Net(torch.nn.Module):
    def __init__(self, functional_groups, num_nodes, hidden_dim, edge_dim, out_dim, atlas=116, dropout=0.5):
        super(Net, self).__init__()
        self.brain_encoding = BrainEncoding(functional_groups=functional_groups, num_nodes=num_nodes, hidden_dim=hidden_dim, atlas=atlas)
        self.node_conv1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)
        self.edge_conv1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False)
        self.node_conv2 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        
    
    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        batch_size = data.num_graphs
        num_nodes = data.num_nodes//batch_size
        num_edges = data.num_edges//batch_size

        data = self.brain_encoding(data)
        n_adj = node_adjacency_matrix(data.edge_index, num_nodes)
        e_adj = edge_adjacency_matrix(n_adj)
        T = transition_matrix(n_adj)
        x, edge_attr = self.node_conv1(data.x, data.edge_attr, e_adj, n_adj, T)        
        x, edge_attr = F.relu(x), F.relu(edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        x, edge_attr = self.edge_conv1(x, edge_attr, e_adj, n_adj, T)
        x, edge_attr = F.relu(x), F.relu(edge_attr)
        x = F.dropout(x, p=self.dropout, training=self.training)
        edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)


        x, edge_attr = self.node_conv2(x, edge_attr, e_adj, n_adj, T)


        x = global_mean_pool(x, data.batch)
        x = self.lin(x)

        return x
    



dataset = ConnectivityDataset(connectivities, labels)
generator = torch.Generator(device=device)


def check_for_nans(tensor, tensor_name):
    if torch.isnan(tensor).any():
        print_with_timestamp(f"NaN detected in {tensor_name}")



# Hyperparameters
n_folds = 10
epochs = 300
batch_size = 1
learning_rate = 0.00001
hidden_dim = 64
dropout = 0.5
patience = 30

test_size = 0.2


n_nodes = 116
edge_dim = 5
out_dim = 4 if dataset_name == 'ppmi' else 2
# Cross-validation setup
kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
# Initialize lists to store metrics for all folds
all_fold_metrics = {'accuracy': [], 'precision': [], 'f1': []}

# Training and evaluation loop
for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    print_with_timestamp(f"Fold {fold + 1}/{n_folds}")
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=test_size, random_state=seed)
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, generator=generator)

    model = Net(functional_groups, n_nodes, hidden_dim, edge_dim, out_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y)
            loss.backward(retain_graph=True)
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                output = model(data)
                val_loss += criterion(output, data.y).item()
        val_loss /= len(val_loader)
        
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
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = output.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(data.y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted')
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