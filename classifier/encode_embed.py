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

gpu_id = 0

device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.set_default_device(device)

dataset_name = 'ppmi'



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

# Define the Dataset class
class ConnectivityDataset(Dataset):
    def __init__(self, connectivities, labels, device='cpu'):
        self.connectivities = connectivities
        self.node_num = len(connectivities[0])
        self.labels = labels
        self.left_indices, self.right_indices = self.get_hemisphere_indices()
        self.device = device

    def get_hemisphere_indices(self):
        left_indices = [i for i in range(self.node_num) if i % 2 == 0]
        right_indices = [i for i in range(self.node_num) if i % 2 != 0]
        return left_indices, right_indices

    def isInter(self,i,j):
        if i in self.left_indices and j in self.right_indices:
            return True
        if i in self.right_indices and j in self.left_indices:
            return True
        return False

    def isIntra(self,i,j):
        if i in self.left_indices and j in self.left_indices:
            return True
        if i in self.right_indices and j in self.right_indices:
            return True
        return False

    def isLeftIntra(self, i,j):
        return i in self.left_indices and j in self.left_indices

    def isRightIntra(self, i, j):
        return i in self.right_indices and j in self.right_indices

    
    def isHomo(self,i,j):
        return i//2 == j//2 and abs(i-j) == 1

    def __len__(self):
        return len(self.connectivities)

    def __getitem__(self, idx):
        connectivity = self.connectivities[idx]
        label = self.labels[idx]
        edge_index = []
        edge_attr = []
        for j in range(connectivity.shape[0]):
            for k in range(j+1, connectivity.shape[0], 1):
                edge_index.append([j, k])
                edge_attr.append([connectivity[j, k], self.isInter(j, k), self.isLeftIntra(j,k), self.isRightIntra(j,k), self.isHomo(j, k)])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
        y = label.type(torch.long)
        data = Data(x=None, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=connectivity.shape[0])
        return data
    

def collate_fn(batch, device='cpu'):
    batch = Batch.from_data_list(batch).to(device)
    return batch



data = torch.load(f'data/{dataset_name}.pth')
connectivities = data['matrix']
labels = data['label']
print(f'Connectivities: {connectivities.shape}, Labels: {labels.shape}')
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

    def forward(self):
        return self.encoding

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
        if self.node_layer:
            multiplier1 = torch.spmm(T, torch.diag((H_e @ self.p.t()).t()[0])) @ T.to_dense().t()
            mask1 = torch.eye(multiplier1.shape[0])
            M1 = mask1 * torch.ones(multiplier1.shape[0]) + (1. - mask1)*multiplier1
            adjusted_A = torch.mul(M1, adj_v.to_dense())
            output = torch.mm(adjusted_A, torch.mm(H_v, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e

        else:
            multiplier2 = torch.spmm(T.t(), torch.diag((H_v @ self.p.t()).t()[0])) @ T.to_dense()
            mask2 = torch.eye(multiplier2.shape[0])
            M3 = mask2 * torch.ones(multiplier2.shape[0]) + (1. - mask2)*multiplier2
            adjusted_A = torch.mul(M3, adj_e.to_dense())
            normalized_adjusted_A = adjusted_A / adjusted_A.max(0, keepdim=True)[0]
            output = torch.mm(normalized_adjusted_A, torch.mm(H_e, self.weight))
            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret
        

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features_v) + ' -> ' \
               + str(self.out_features_v) + '), (' \
               + str(self.in_features_e) + ' -> ' \
               + str(self.out_features_e) + ')'

# def fully_connected_edge_adjacency_matrix(n):
#     num_edges = int((n * (n - 1)) / 2)
#     A = np.ones((num_edges, num_edges))  # Create an num_edges x num_edges matrix filled with ones
#     np.fill_diagonal(A, 0)  # Set the diagonal to zeros
#     return torch.tensor(A, dtype=torch.float)

# def fully_connected_node_adjacency_matrix(n):
#     A = np.ones((n, n))  # Create an n x n matrix filled with ones
#     np.fill_diagonal(A, 0)  # Set the diagonal to zeros
#     return torch.tensor(A, dtype=torch.float)

# def create_T_matrix_pyg(num_nodes, edge_index):
#     num_edges = edge_index.shape[1]
#     T = torch.zeros((num_nodes, num_edges), dtype=torch.float)
    
#     for edge_idx in range(num_edges):
#         node1 = edge_index[0, edge_idx]
#         node2 = edge_index[1, edge_idx]
#         T[node1, edge_idx] = 1
#         T[node2, edge_idx] = 1
#     return T

class Net(torch.nn.Module):
    def __init__(self, functional_groups, num_nodes, hidden_dim, edge_dim, out_dim, atlas=116, dropout=0.5, device='cpu'):
        super(Net, self).__init__()
        self.brain_encoding = BrainEncoding(functional_groups=functional_groups, num_nodes=num_nodes, hidden_dim=hidden_dim, atlas=atlas)
        self.node_conv1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)
        self.edge_conv1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False)
        self.node_conv2 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.device = device

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.brain_encoding().to(self.device)
        T = self.create_T_matrix_pyg(x.shape[0], edge_index).to(self.device)
        e_adj = self.fully_connected_edge_adjacency_matrix(x.shape[0]).to(self.device)
        n_adj = self.fully_connected_node_adjacency_matrix(x.shape[0]).to(self.device)
        x, edge_attr = self.node_conv1(x, edge_attr, e_adj, n_adj, T)
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
        return F.log_softmax(x, dim=1)
    
    def fully_connected_edge_adjacency_matrix(self,n):
        num_edges = int((n * (n - 1)) / 2)
        A = np.ones((num_edges, num_edges))  # Create an num_edges x num_edges matrix filled with ones
        np.fill_diagonal(A, 0)  # Set the diagonal to zeros
        return torch.tensor(A, dtype=torch.float)

    def fully_connected_node_adjacency_matrix(self,n):
        A = np.ones((n, n))  # Create an n x n matrix filled with ones
        np.fill_diagonal(A, 0)  # Set the diagonal to zeros
        return torch.tensor(A, dtype=torch.float)

    def create_T_matrix_pyg(self,num_nodes, edge_index):
        num_edges = edge_index.shape[1]
        T = torch.zeros((num_nodes, num_edges), dtype=torch.float)
        
        for edge_idx in range(num_edges):
            node1 = edge_index[0, edge_idx]
            node2 = edge_index[1, edge_idx]
            T[node1, edge_idx] = 1
            T[node2, edge_idx] = 1
        return T

dataset = ConnectivityDataset(connectivities, labels, device=device)
generator = torch.Generator(device=device)


# Hyperparameters
n_folds = 10
epochs = 300
batch_size = 1
learning_rate = 0.001
hidden_dim = 512
dropout = 0.5
patience = 30
seed = 10
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
    print(f"Fold {fold + 1}/{n_folds}")
    train_data = torch.utils.data.Subset(dataset, train_index)
    test_data = torch.utils.data.Subset(dataset, test_index)

    # Split training data into train and validation sets for early stopping
    train_indices, val_indices = train_test_split(range(len(train_data)), test_size=test_size, random_state=seed)
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, device), generator=generator)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, device), generator=generator)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, device), generator=generator)

    model = Net(functional_groups, n_nodes, hidden_dim, edge_dim, out_dim, dropout=dropout, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item()}, Val Loss: {val_loss}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
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
    print(f"Fold {fold + 1} Metrics: Accuracy: {acc}, Precision: {precision}, F1 Score: {f1}")

    # Store metrics for the current fold
    all_fold_metrics['accuracy'].append(acc)
    all_fold_metrics['precision'].append(precision)
    all_fold_metrics['f1'].append(f1)

    # Print average metrics until the latest fold
    avg_accuracy = np.mean(all_fold_metrics['accuracy'])
    avg_precision = np.mean(all_fold_metrics['precision'])
    avg_f1 = np.mean(all_fold_metrics['f1'])
    print(f"Average Metrics until Fold {fold + 1}: Accuracy: {avg_accuracy}, Precision: {avg_precision}, F1 Score: {avg_f1}")

# Compute and print final metrics after cross-validation
final_accuracy = np.mean(all_fold_metrics['accuracy'])
final_precision = np.mean(all_fold_metrics['precision'])
final_f1 = np.mean(all_fold_metrics['f1'])

print("Training completed.")
print(f"Final Metrics:\nAccuracy: {final_accuracy}\nPrecision: {final_precision}\nF1 Score: {final_f1}")