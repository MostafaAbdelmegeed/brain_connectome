
import torch
import math
import numpy as np
from torch.nn import Module, Parameter, Dropout, LeakyReLU, Sequential, ReLU
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, GATv2Conv, BatchNorm, LayerNorm, GCNConv, Linear, ResGatedGraphConv, GINConv, GINEConv
from torch_geometric.nn import MLP as pyg_MLP
from encoding import yeo_network
import torch.nn as nn
import torch.nn.functional as F
import datetime


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

# class OGBrainEncodeEmbed(MessagePassing):
#     def __init__(self, functional_groups, hidden_dim, edge_dim, n_roi=116, embedding_dim=16, dropout=0.7):
#         super(OGBrainEncodeEmbed, self).__init__()
#         self.functional_groups = functional_groups
#         self.n_groups = len(functional_groups)
#         self.hidden_dim = hidden_dim
#         self.num_nodes = n_roi
#         self.embedding_dim = embedding_dim
        
#         # Define a learnable embedding layer
#         self.group_embedding = nn.Embedding(self.n_groups, self.embedding_dim)
        
#         # Linear layer for combining node features and group embeddings
#         self.linear = Linear(n_roi + self.embedding_dim, hidden_dim)
#         self.relu = LeakyReLU()
#         self.dropout = Dropout(p=dropout)  # Adding dropout here
        
#         # Other layers remain the same
#         self.coembed_1 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=False, dropout=dropout)
#         self.coembed_2 = AlternateConvolution(in_features_v=hidden_dim, out_features_v=hidden_dim, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=True, dropout=dropout)

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def create_encoding(self):
#         encoding = torch.zeros((self.num_nodes, self.embedding_dim), device=self.device)  # Initialize encoding for nodes

#         # Loop through the functional groups and assign the embeddings
#         for group_id, nodes in enumerate(self.functional_groups.values()):
#             # Fetch the learnable embedding for the current group
#             group_embedding = self.group_embedding(torch.tensor(group_id, device=self.device))
            
#             # Assign the same embedding to all nodes in the group
#             for node in nodes:
#                 encoding[node] = group_embedding
                
#         return encoding


#     def get_group_encoding(self, group):
#         encoding = torch.zeros(self.n_groups)
#         encoding[hash(group) % self.n_groups] = 1
#         return encoding

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.group_embedding.weight)  # Use Xavier initialization for embeddings

#     def forward(self, data):
#         x = data.x
#         edge_attr = data.edge_attr
#         n_adj = data.node_adj
#         e_adj = data.edge_adj
#         transition = data.transition

#         batch_size = data.num_graphs
#         num_nodes = x.size(0)

#         # Create an expanded embedding for each node based on its functional group
#         expanded_encoding = torch.zeros((num_nodes, self.embedding_dim), device=x.device)
#         for group_id, nodes in enumerate(self.functional_groups.values()):
#             group_embedding = self.group_embedding(torch.tensor(group_id, device=x.device))
#             expanded_encoding[nodes] = group_embedding

#         # Concatenate node features with functional group embeddings
#         if x is not None:
#             x = torch.cat([x, expanded_encoding], dim=-1)
        
#         x = self.dropout(x)
#         # Pass the concatenated features through a linear layer
#         x = self.relu(self.linear(x))
        
#         # Perform co-embedding operations
#         x, edge_attr = self.coembed_1(x, edge_attr, e_adj, n_adj, transition)
#         x = x.reshape(data.num_nodes, -1, x.size(2)).squeeze(1)
#         edge_attr = edge_attr.reshape(data.num_edges, -1, edge_attr.size(2)).squeeze(1)
#         x, edge_attr = self.coembed_2(x, edge_attr, e_adj, n_adj, transition)
#         x = x.reshape(data.num_nodes, -1, x.size(2)).squeeze(1)
#         edge_attr = edge_attr.reshape(data.num_edges, -1, edge_attr.size(2)).squeeze(1)

#         return x, edge_attr

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
            ReLU(),
            Linear(hidden_dim, hidden_dim)
        )
        # GINEConv layer
        self.conv = GINEConv(nn=self.mlp, edge_dim=edge_dim)
        self.relu = ReLU()
        self.dropout = Dropout(p=self.dropout_rate)
        self.bn = BatchNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize group embeddings
        torch.nn.init.xavier_uniform_(self.group_embedding.weight)
        # Initialize MLP layers in GINEConv
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        # Initialize LayerNorm
        if hasattr(self.bn, 'reset_parameters'):
            self.bn.reset_parameters()

    def forward(self, data):
        x = data.x  # Node features [N, in_channels]
        edge_index = data.edge_index  # Edge indices [2, E]
        edge_attr = data.edge_attr  # Edge features [E, edge_dim]
        num_nodes = x.size(0)
        # Create functional group embeddings for each node
        group_ids = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        for group_id, nodes in enumerate(self.functional_groups.values()):
            group_ids[nodes] = group_id
        # Use embedding layer to get group embeddings
        expanded_encoding = self.group_embedding(group_ids)
        # Concatenate node features with functional group embeddings
        x = torch.cat([x, expanded_encoding], dim=-1)  # [N, in_channels + embedding_dim]
        x = self.dropout(x)
        # Map concatenated features to hidden_dim
        x = self.relu(Linear(x.size(-1), self.hidden_dim).to(x.device)(x))
        x = self.bn(x)
        x = self.dropout(x)
        # Apply GINEConv
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        # Apply activation and normalization
        x = self.relu(x)
        return x, edge_attr

class AlternateConvolution(Module):
    def __init__(self, in_features_v, out_features_v, in_features_e, out_features_e, bias=True, node_layer=True, dropout=0.7):
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
        # Initialize self.p
        torch.nn.init.xavier_uniform_(self.p)


    def forward(self, H_v, H_e, adj_e, adj_v, T):
        device = H_v.device  # Ensure all operations are on the same device
        batch_size = adj_v.shape[0]//adj_v.shape[1]
        H_v = H_v.view(batch_size, -1, self.in_features_v)
        H_e = H_e.view(batch_size, -1, self.in_features_e).type(torch.float32)
        adj_e = adj_e.to_dense().view(batch_size, -1, adj_e.size(1), adj_e.size(1)).squeeze(1)
        adj_v = adj_v.to_dense().view(batch_size, -1, adj_v.size(1), adj_v.size(1)).squeeze(1)
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
        # Attention mechanism: projecting node features into hidden_dim space and then to scalar weights
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_dim),  # Project node features to hidden_dim
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)  # Scalar attention weight
        )

    def forward(self, x, batch, mask=None):
        """
        x: Node features of shape [total_num_nodes, in_features]
        batch: Tensor of shape [total_num_nodes] indicating the graph index for each node
        mask: Optional mask for ignoring certain nodes (if provided)
        """
        # Compute attention weights per node
        weights = self.project(x).squeeze(-1)  # Shape: [total_num_nodes]
        
        if mask is not None:
            weights = weights.masked_fill(mask == 0, -1e9)  # Mask out invalid nodes
            
        # Apply softmax per graph (normalize within each graph)
        weights = F.softmax(weights, dim=0)  # Shape: [total_num_nodes]
        
        # Multiply node features by their attention weights
        x = x * weights.unsqueeze(-1)  # Shape: [total_num_nodes, in_features]
        
        # Use scatter_add to sum the weighted features per graph
        out = torch.zeros(batch.max() + 1, x.size(1), device=x.device)
        out = out.scatter_add(0, batch.unsqueeze(-1).expand_as(x), x)
        
        return out, weights

# # GCN
# class BrainBlock(Module):
#     def __init__(self, in_features, out_features, edge_dim=5, heads=1, dropout=0.7):
#         super(BrainBlock, self).__init__()
#         self.conv = GCNConv(in_features, out_features)
#         self.bn = BatchNorm(out_features)  
#         self.dropout = Dropout(p=dropout)
#         self.relu = LeakyReLU()
#         self.reset_parameters()

#     def reset_parameters(self):
#         torch.nn.init.xavier_uniform_(self.conv.lin.weight)  # Apply Xavier initialization to GCNConv weights
#         if hasattr(self.conv, 'bias') and self.conv.lin.bias is not None:
#             torch.nn.init.zeros_(self.conv.lin.bias)  # Initialize bias to zero if it exists
        
#     def forward(self, x, edge_index, edge_attr):
#         edge_weight = torch.abs(edge_attr[:, 0])
#         x = self.conv(x, edge_index, edge_weight)
#         x = self.relu(x)
#         x = self.bn(x)
#         x = self.dropout(x)
#         return x

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

class BrainNetGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, edge_dim=5, dropout=0.7):
        super(BrainNetGIN, self).__init__()
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
                        ReLU(),
                        Linear(hidden_channels, hidden_channels)
                    ), edge_dim=edge_dim
                )
            )
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = Dropout(p=dropout)
        self.relu = ReLU()
        self.reset_parameters()
        self.bn = BatchNorm(hidden_channels)

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
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.bn(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = global_add_pool(x, data.batch)
        x = self.lin1(self.relu(x))
        x = self.lin2(x)
        return x
    
# class GCNBlock(torch.nn.Module):
#     def __init__(self, in_features, out_features, dropout=0.7):
#         super(GCNBlock, self).__init__()
#         self.conv = GCNConv(in_features, out_features)
#         self.relu = LeakyReLU()
#         self.bn = BatchNorm(out_features)
#         self.dropout = Dropout(p=dropout)
        
#     def forward(self, x, edge_index, edge_weight):
#         edge_weight = (edge_weight + 1) / 2  # Normalize from [-1, 1] to [0, 1]
#         x = self.conv(x, edge_index, edge_weight)
#         x = self.relu(x)
#         x = self.bn(x)
#         x = self.dropout(x)
#         return x

class BrainNetGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, edge_dim=5, dropout=0.7):
        super(BrainNetGCN, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.bn_node = BatchNorm(hidden_channels)
        # self.bn_edge = BatchNorm(edge_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.edge_weight_transform = Linear(edge_dim, 1)  # Learnable transformation from 5D to 1D
        self.dropout = dropout

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        x = self.bn_node(x)
        # # edge_attr = self.bn_edge(edge_attr)
        # edge_weight = torch.abs(edge_attr[:, 0])
        # Learnable weighted sum for edge attributes
        edge_weight = F.relu(self.edge_weight_transform(edge_attr).squeeze())
        edge_index = data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x
    
# class GATBlock(torch.nn.Module):
#     def __init__(self, in_features, out_features, heads=1, dropout=0.5, edge_dim=5):
#         super(GATBlock, self).__init__()
#         self.conv = GATv2Conv(in_features, out_features, heads=heads, dropout=dropout, edge_dim=edge_dim, concat=False)
#         self.bn = BatchNorm(out_features)
#         self.relu = LeakyReLU()
        
#     def forward(self, x, edge_index, edge_attr):
#         x = self.conv(x, edge_index, edge_attr)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

class BrainNetGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, heads=1, edge_dim=5, dropout=0.5):
        super(BrainNetGAT, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim, concat=False))
        self.pooling = AttentionPooling(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        for _, layer in enumerate(self.layers):
            x = layer(x, data.edge_index, edge_attr)
            x = F.relu(x)
        x, attn_weights = self.pooling(x, data.batch)
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x
    

class BrainNetAltGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, heads=1, edge_dim=5, dropout=0.5):
        super(BrainNetAltGCN, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups=functional_groups, hidden_dim=hidden_channels, edge_dim=edge_dim, n_roi=116)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(AlternateConvolution(in_features_v=hidden_channels, out_features_v=hidden_channels, in_features_e=edge_dim, out_features_e=edge_dim, node_layer=(len(self.layers)%2==0), dropout=dropout))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_attr = self.encemb(data)
        n_adj, e_adj, transition = data.node_adj, data.edge_adj, data.transition
        for _, layer in enumerate(self.layers):
            x, edge_attr = layer(x, edge_attr, e_adj, n_adj, transition)
            x, edge_attr = F.relu(x), F.relu(edge_attr)
            x, edge_attr = F.dropout(x, p=self.dropout, training=self.training), F.dropout(edge_attr, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch)
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x).squeeze(1)
        return x
    
    


def get_model(args, edge_dim=5):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    out_channels = 4 #
    dropout = args.dropout
    heads = args.heads
    act = ReLU()
    groups = yeo_network()
    if model_name == 'gin':
        return BrainNetGIN(in_channels=3, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups, edge_dim=edge_dim)
    elif model_name == 'gcn':
        return BrainNetGCN(in_channels=3, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups, edge_dim=edge_dim)
    elif model_name == 'gat':
        return BrainNetGAT(in_channels=3, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups, edge_dim=edge_dim, heads=heads) 
    elif model_name == 'alt_gcn':
        return BrainNetAltGCN(in_channels=3, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups, edge_dim=edge_dim)
    else:
        raise ValueError(f'Unknown model name: {model_name}')