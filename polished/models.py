
import torch
import math
import numpy as np
from torch.nn import Module, Parameter, Dropout, LeakyReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, GATv2Conv, BatchNorm, GCNConv, Linear, ResGatedGraphConv, Sequential, GINConv
from torch_geometric.nn.models import GCN, GAT
from encoding import yeo_network
import torch.nn as nn
import torch.nn.functional as F


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
        expanded_encoding = self.encoding.repeat(batch_size, 1).to(x.device)

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
    def __init__(self, in_features, out_features, edge_dim, heads=1, dropout=0.7):
        super(BrainBlock, self).__init__()
        self.gat = GATv2Conv(in_features, out_features, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.res = ResGatedGraphConv(out_features*heads, out_features, edge_dim=edge_dim)
        self.bn = BatchNorm(out_features)
        self.dropout = Dropout(p=dropout)
        self.relu = LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.gat(x, edge_index, edge_attr)
        x = self.res(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class BrainNet(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, out_channels, functional_groups=None, edge_dim=5, heads=1, dropout=0.7):
        super(BrainNet, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(BrainBlock(hidden_channels, hidden_channels, edge_dim, heads=heads, dropout=dropout))
        self.gin = GINConv(Sequential('x', [(Linear(hidden_channels, hidden_channels), 'x -> x'), LeakyReLU(inplace=True), (Linear(hidden_channels, hidden_channels), 'x -> x')]), train_eps=True)
        self.fc = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        for _, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, edge_attr)
        x = self.gin(x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x
    

    


def get_model(args):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    out_channels = 4 #
    dropout = args.dropout
    heads = args.heads
    act = LeakyReLU()
    groups = yeo_network()
    if model_name == 'brain':
        return BrainNet(hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, heads=heads, dropout=dropout, functional_groups=groups)
    elif model_name == 'gcn':
        return GCN(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, act=act, dropout=dropout)
    elif model_name == 'gat':
        return GAT(in_channels=116, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, act=act, dropout=dropout)
    else:
        raise ValueError(f'Unknown model name: {model_name}')