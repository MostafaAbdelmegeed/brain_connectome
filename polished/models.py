
import torch
import math
import numpy as np
from torch.nn import Module, Parameter, Dropout, LeakyReLU, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GATv2Conv, BatchNorm, LayerNorm, GCNConv, Linear, ResGatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import MLP as pyg_MLP
from encoding import yeo_network
import torch.nn as nn
import torch.nn.functional as F
import datetime


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")


class BrainEncodeEmbed(torch.nn.Module):
    def __init__(self, functional_groups, hidden_dim, edge_dim, n_roi=116, embedding_dim=16, dropout=0.7):
        super(BrainEncodeEmbed, self).__init__()
        self.functional_groups = functional_groups
        self.n_groups = len(functional_groups)
        self.hidden_dim = hidden_dim
        self.num_nodes = n_roi
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout

        # Define a learnable embedding layer for functional groups
        self.group_embedding = torch.nn.Embedding(self.n_groups, self.embedding_dim)

        # MLP for GINEConv (node feature transformation)
        self.mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            LeakyReLU(),
            Linear(hidden_dim, hidden_dim)
        )

        # GINEConv layer
        self.conv = GINEConv(nn=self.mlp)

        # Edge encoder to map edge features to the same dimensionality as node features
        self.edge_encoder = Linear(edge_dim, hidden_dim)

        self.relu = LeakyReLU()
        self.dropout = Dropout(p=self.dropout_rate)
        self.ln = LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize group embeddings
        torch.nn.init.xavier_uniform_(self.group_embedding.weight)
        # Initialize edge encoder
        torch.nn.init.xavier_uniform_(self.edge_encoder.weight)
        torch.nn.init.zeros_(self.edge_encoder.bias)
        # Initialize MLP layers in GINEConv
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        # Initialize LayerNorm
        if hasattr(self.ln, 'reset_parameters'):
            self.ln.reset_parameters()

    def forward(self, data):
        x = data.x  # Node features [N, in_channels]
        edge_index = data.edge_index  # Edge indices [2, E]
        edge_attr = data.edge_attr  # Edge features [E, edge_dim]

        num_nodes = x.size(0)

        # Create functional group embeddings for each node
        expanded_encoding = torch.zeros((num_nodes, self.embedding_dim), device=x.device)
        for group_id, nodes in enumerate(self.functional_groups.values()):
            group_embedding = self.group_embedding(torch.tensor(group_id, device=x.device))
            expanded_encoding[nodes] = group_embedding

        # Concatenate node features with functional group embeddings
        x = torch.cat([x, expanded_encoding], dim=-1)  # [N, in_channels + embedding_dim]
        x = self.dropout(x)

        # Map concatenated features to hidden_dim
        x = self.relu(Linear(x.size(-1), self.hidden_dim).to(x.device)(x))

        # Encode edge attributes to match hidden_dim
        edge_emb = self.relu(self.edge_encoder(edge_attr))

        # Apply GINEConv
        x = self.conv(x, edge_index, edge_attr=edge_emb)

        # Apply activation and normalization
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)

        return x, edge_attr

class AlternateConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True, dropout=0.7):
        super(AlternateConvolution, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.dropout = Dropout(p=dropout)  # Adding dropout here
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
        if self.node_layer:
            torch.nn.init.xavier_uniform_(self.weight)  # Xavier for node features
        else:
            torch.nn.init.xavier_uniform_(self.weight)  # Xavier for edge features
        
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # Initialize bias to zero
        # Initialize self.p
        torch.nn.init.xavier_uniform_(self.p)


    def forward(self, H_v, H_e, adj_e, adj_v, T):
        device = H_v.device  # Ensure all operations are on the same device
        batch_size = adj_v.shape[0]//adj_v.shape[1]
        H_v = H_v.view(batch_size, -1, self.in_features_v)
        H_e = H_e.view(batch_size, -1, self.in_features_e).type(torch.float32)
        adj_e = adj_e.view(batch_size, -1, adj_e.size(1), adj_e.size(1)).squeeze(1)
        adj_v = adj_v.view(batch_size, -1, adj_v.size(1), adj_v.size(1)).squeeze(1)
        T = T.to_dense().view(batch_size, -1, T.size(0)//batch_size, T.size(1)).squeeze(1)
        if self.node_layer:
            H_e_p = H_e @ self.p.t()  # Adapted for batch
            diag_values = H_e_p.transpose(1, 2)
            diag_matrix = torch.diag_embed(diag_values).transpose(1, 2).squeeze(2)
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
            multiplier2 = torch.bmm(T.transpose(1, 2), diag_matrix) @ T
            mask2 = torch.eye(multiplier2.size(1), device=device).unsqueeze(0).repeat(multiplier2.size(0), 1, 1)
            M3 = mask2 * torch.ones_like(multiplier2) + (1. - mask2) * multiplier2
            adjusted_A = torch.mul(M3, adj_e)
            normalized_adjusted_A = adjusted_A / (adjusted_A.max(2, keepdim=True)[0] + 1e-10)
            weight_repeated = self.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            output = torch.bmm(normalized_adjusted_A, torch.bmm(H_e, weight_repeated))
            # Apply dropout after the edge layer processing
            output = self.dropout(output)
            if self.bias is not None:
                output += self.bias
            return H_v, output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features_v) + ' -> ' + str(self.out_features_v) + '), (' + str(self.in_features_e) + ' -> ' + str(self.out_features_e) + ')'


class AttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, x, batch, mask=None):
        # x: shape [total_num_nodes, in_features]
        # batch: shape [total_num_nodes] (indicating the graph index for each node)
        
        # Compute attention weights
        weights = self.project(x).squeeze(-1)  # Shape: [total_num_nodes]
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)  # Apply mask
        weights = F.softmax(weights, dim=0)  # Normalize across nodes
        
        # Compute the weighted sum of node features per graph
        x = x * weights.unsqueeze(-1)  # Shape: [total_num_nodes, in_features]
        x = torch.zeros_like(x).scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        
        # Pooling using the scatter_add result based on the batch index
        x = torch.zeros(batch.max() + 1, x.size(1), device=x.device).scatter_add_(
            0, batch.unsqueeze(-1).expand_as(x), x
        )  # Shape: [num_graphs, in_features]
        
        return x, weights

# GCN
class BrainBlock(Module):
    def __init__(self, in_features, out_features, edge_dim=5, heads=1, dropout=0.7):
        super(BrainBlock, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        self.ln = LayerNorm(out_features)  
        self.dropout = Dropout(p=dropout)
        self.relu = LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.conv.lin.weight)  # Apply Xavier initialization to GCNConv weights
        if hasattr(self.conv, 'bias') and self.conv.lin.bias is not None:
            torch.nn.init.zeros_(self.conv.lin.bias)  # Initialize bias to zero if it exists
        
    def forward(self, x, edge_index, edge_attr):
        edge_weight = torch.abs(edge_attr[:, 0])
        x = self.conv(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x

# # GAT
# class BrainBlock(Module):
#     def __init__(self, in_features, out_features, edge_dim, dropout=0.7, heads=1):
#         super(BrainBlock, self).__init__()
#         self.conv = GATv2Conv(in_features, out_features, dropout=dropout, edge_dim=edge_dim, heads=heads, concat=False)
#         self.ln = LayerNorm(out_features)  
#         self.dropout = Dropout(p=dropout)
#         self.relu = LeakyReLU()
#         self.reset_parameters()

#     def reset_parameters(self):
#         self.conv.reset_parameters()
#         if hasattr(self.ln, 'reset_parameters'):
#             self.ln.reset_parameters()
        
#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv(x, edge_index, edge_attr)
#         x = self.relu(x)
#         x = self.ln(x)
#         x = self.dropout(x)
#         return x

# # GIN
# class BrainBlock(Module):
#     def __init__(self, in_features, out_features, edge_dim, dropout=0.7, heads=1, num_layers=2):
#         super(BrainBlock, self).__init__()
#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             mlp = pyg_MLP([in_channels, out_features, out_features])
#             self.convs.append(GINConv(nn=mlp, train_eps=False))
#             in_channels = out_features

#         self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5)

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = global_add_pool(x, data.batch)
#         return self.mlp(x), x

#     def reset_parameters(self):
#         self.conv.reset_parameters()
#         if hasattr(self.ln, 'reset_parameters'):
#             self.ln.reset_parameters()
        
#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv(x, edge_index, edge_attr)
#         x = self.relu(x)
#         x = self.ln(x)
#         x = self.dropout(x)
#         return x


# class BrainGIN(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, functional_groups=None, edge_dim=5, dropout=0.5, n_roi=116):
#         super().__init__()
#         self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=n_roi)
#         self.convs = torch.nn.ModuleList()
#         for _ in range(num_layers):
#             mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels])
#             self.convs.append(GINConv(nn=mlp, train_eps=False))
#             in_channels = hidden_channels
#         self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#         edge_weight = torch.abs(edge_attr[:, 0])
#         x, edge_attr = self.encemb()
#         for conv in self.convs:
#             x = F.relu(conv(x, edge_index))
#         x = global_add_pool(x, data.batch)
#         return self.mlp(x)

class BrainNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, edge_dim=5, dropout=0.7):
        super(BrainNet, self).__init__()
        self.encemb = BrainEncodeEmbed(
            functional_groups=functional_groups,
            hidden_dim=hidden_channels,
            edge_dim=edge_dim,
            n_roi=116,
            dropout=dropout
        )
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            # You can use additional GINEConv layers or other convolutional layers
            self.layers.append(
                GINEConv(
                    nn=Sequential(
                        Linear(hidden_channels, hidden_channels),
                        LeakyReLU(),
                        Linear(hidden_channels, hidden_channels)
                    )
                )
            )
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = Dropout(p=dropout)
        self.relu = LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.encemb.reset_parameters()
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                for param in layer.parameters():
                    if param.dim() > 1:
                        torch.nn.init.xavier_uniform_(param)
                    else:
                        torch.nn.init.zeros_(param)
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        edge_index = data.edge_index
        edge_emb = self.encemb.edge_encoder(edge_attr)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_emb)
            x = self.relu(x)
            x = self.dropout(x)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(self.relu(x))
        x = self.lin2(x)
        return x
    
class GCNBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, dropout=0.7):
        super(GCNBlock, self).__init__()
        self.conv = GCNConv(in_features, out_features)
        self.relu = LeakyReLU()
        self.ln = LayerNorm(out_features)
        self.dropout = Dropout(p=dropout)
        
    def forward(self, x, edge_index, edge_weight):
        edge_weight = (edge_weight + 1) / 2  # Normalize from [-1, 1] to [0, 1]
        x = self.conv(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.ln(x)
        x = self.dropout(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.7):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr[:, 0]
        edge_weight = torch.abs(edge_weight)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x.float(), edge_index, edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch) 
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x
    
class GATBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, heads=1, dropout=0.5, edge_dim=5):
        super(GATBlock, self).__init__()
        self.conv = GATv2Conv(in_features, out_features, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.bn = BatchNorm(out_features*heads)
        self.relu = LeakyReLU()
        
    def forward(self, x, edge_index, edge_attr):
        x = self.conv(x, edge_index, edge_attr)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, functional_groups=None, heads=1, edge_dim=5, dropout=0.5):
        super(GAT, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            if len(self.layers) == 0:
                self.layers.append(GATBlock(hidden_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim))
            else:
                self.layers.append(GATBlock(hidden_channels*heads, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim))
        self.fc2 = Linear(hidden_channels*heads, out_channels)

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        for _, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, edge_attr)
        x = global_mean_pool(x, data.batch)
        x = self.fc2(x)
        return x
    


def get_model(args, edge_dim=5):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    out_channels = 4 #
    dropout = args.dropout
    heads = args.heads
    act = LeakyReLU()
    groups = yeo_network()
    if model_name == 'brain':
        return BrainNet(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups, edge_dim=edge_dim)
    elif model_name == 'gcn':
        return GCN(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout)
    elif model_name == 'gat':
        return GAT(hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, functional_groups=groups, dropout=dropout, heads=heads, edge_dim=edge_dim)
    else:
        raise ValueError(f'Unknown model name: {model_name}')