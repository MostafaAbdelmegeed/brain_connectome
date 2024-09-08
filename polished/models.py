
import torch
import math
import numpy as np
from torch.nn import Module, Parameter, Dropout, LeakyReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, GATv2Conv, BatchNorm, LayerNorm, GCNConv, Linear, ResGatedGraphConv, Sequential, GINConv
from encoding import yeo_network
import torch.nn as nn
import torch.nn.functional as F


class BrainEncodeEmbed(MessagePassing):
    def __init__(self, functional_groups, hidden_dim, edge_dim, n_roi=116, embedding_dim=16):
        super(BrainEncodeEmbed, self).__init__()
        self.functional_groups = functional_groups
        self.n_groups = len(functional_groups)
        self.hidden_dim = hidden_dim
        self.num_nodes = n_roi
        self.embedding_dim = embedding_dim
        
        # Define a learnable embedding layer
        self.group_embedding = nn.Embedding(self.n_groups, self.embedding_dim)
        
        # Linear layer for combining node features and group embeddings
        self.linear = Linear(n_roi + self.embedding_dim, hidden_dim)
        self.relu = LeakyReLU()
        
        # Other layers remain the same
        # self.coembed_1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False)
        self.coembed_2 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def create_encoding(self):
        encoding = torch.zeros((self.num_nodes, self.embedding_dim), device=self.device)  # Initialize encoding for nodes

        # Loop through the functional groups and assign the embeddings
        for group_id, nodes in enumerate(self.functional_groups.values()):
            # Fetch the learnable embedding for the current group
            group_embedding = self.group_embedding(torch.tensor(group_id, device=self.device))
            
            # Assign the same embedding to all nodes in the group
            for node in nodes:
                encoding[node] = group_embedding
                
        return encoding


    def get_group_encoding(self, group):
        encoding = torch.zeros(self.n_groups)
        encoding[hash(group) % self.n_groups] = 1
        return encoding

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.group_embedding.weight)  # Use Xavier initialization for embeddings

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        n_adj = data.node_adj
        e_adj = data.edge_adj
        transition = data.transition

        batch_size = data.num_graphs
        num_nodes = x.size(0)

        # Create an expanded embedding for each node based on its functional group
        expanded_encoding = torch.zeros((num_nodes, self.embedding_dim), device=x.device)
        for group_id, nodes in enumerate(self.functional_groups.values()):
            group_embedding = self.group_embedding(torch.tensor(group_id, device=x.device))
            expanded_encoding[nodes] = group_embedding

        # Concatenate node features with functional group embeddings
        if x is not None:
            x = torch.cat([x, expanded_encoding], dim=-1)
        
        # Pass the concatenated features through a linear layer
        x = self.relu(self.linear(x))
        
        # Perform co-embedding operations
        # x, edge_attr = self.coembed_1(x, edge_attr, e_adj, n_adj, transition)
        # x = x.reshape(data.num_nodes, -1, x.size(2)).squeeze(1)
        # edge_attr = edge_attr.reshape(data.num_edges, -1, edge_attr.size(2)).squeeze(1)
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
        if self.node_layer:
            torch.nn.init.xavier_uniform_(self.weight)  # Xavier for node features
        else:
            torch.nn.init.xavier_uniform_(self.weight)  # Xavier for edge features
        
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # Initialize bias to zero


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

class BrainBlock(Module):
    def __init__(self, in_features, out_features, edge_dim, dropout=0.7):
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

class BrainNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, edge_dim=5, heads=1, dropout=0.7):
        super(BrainNet, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            if len(self.layers) == 0:
                self.layers.append(BrainBlock(in_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout))
            else:
                self.layers.append(BrainBlock(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()  # Reset parameters for each BrainBlock
        torch.nn.init.xavier_uniform_(self.lin1.weight)  # Apply Xavier initialization to Linear layer
        torch.nn.init.zeros_(self.lin1.bias)  # Initialize bias to zero
        torch.nn.init.xavier_uniform_(self.lin2.weight)  # Apply Xavier initialization to Linear layer
        torch.nn.init.zeros_(self.lin2.bias)  # Initialize bias to zero

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        x, edge_attr = data.x, data.edge_attr
        for _, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, edge_attr)
        x = global_mean_pool(x, data.batch)
        x = self.lin1(F.leaky_relu(x))
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
        return BrainNet(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, heads=heads, dropout=dropout, functional_groups=groups, edge_dim=edge_dim)
    elif model_name == 'gcn':
        return GCN(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout)
    elif model_name == 'gat':
        return GAT(hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, functional_groups=groups, dropout=dropout, heads=heads, edge_dim=edge_dim)
    else:
        raise ValueError(f'Unknown model name: {model_name}')