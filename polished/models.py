
import torch
from torch.nn import ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, Linear, GINConv
from torch_geometric.nn import MLP as pyg_MLP
from encoding import yeo_network
import torch.nn.functional as F
import datetime


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")

class BrainEncodeEmbed(torch.nn.Module):
    def __init__(self, functional_groups, hidden_dim, edge_weight_transform=False):
        super(BrainEncodeEmbed, self).__init__()
        self.functional_groups = functional_groups
        self.n_groups = len(functional_groups)
        self.hidden_dim = hidden_dim
        self.edge_weight_transform = Linear(-1, 1) if edge_weight_transform else None
        # Define a learnable embedding layer for functional groups
        self.group_embedding = torch.nn.Embedding(self.n_groups, 2)
        self.hemisphere_embedding = torch.nn.Embedding(2, 2)
        
    def reset_parameters(self):
        # Initialize group embeddings
        torch.nn.init.xavier_uniform_(self.group_embedding.weight)
        torch.nn.init.xavier_uniform_(self.hemisphere_embedding.weight)
        if self.edge_weight_transform:
            torch.nn.init.xavier_uniform_(self.edge_weight_transform.weight)
        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.zeros_(self.lin1.bias)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.zeros_(self.lin2.bias)

    def forward(self, data):
        x = data.x
        # Create functional group embeddings for each node
        edge_attr = data.edge_attr.float()  # Edge features [E, edge_dim]
        # Transform multi-dimensional edge attributes into scalar edge weight
        if self.edge_weight_transform:
            edge_attr = torch.abs(self.edge_weight_transform(edge_attr).squeeze())

        hemisphere_indicator = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hemisphere_indicator[1::2] = 1  # Assuming alternating left (even indices) and right (odd indices)
        hemisphere_embeddings = self.hemisphere_embedding(hemisphere_indicator)
        
        # Dynamically create group_ids based on actual batch size
        # Use embedding layer to get group embeddings
        group_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for group_id, nodes in enumerate(self.functional_groups.values()):
            group_ids[nodes] = group_id
        # Use embedding layer to get group embeddings
        functional_encoding = self.group_embedding(group_ids)
        # Concatenate node features with functional group embeddings
        x = torch.cat([x, functional_encoding, hemisphere_embeddings], dim=-1)  # [N, in_channels + embedding_dim]
        return x, edge_attr


class BaselineGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(BaselineGIN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
            self.layers.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        self.relu = ReLU()

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        for layer in self.layers:
            x = self.relu(layer(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x)
    


class BrainNetGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5):
        super(BrainNetGIN, self).__init__()
        self.encemb = BrainEncodeEmbed(
            functional_groups=functional_groups,
            hidden_dim=hidden_channels
        )
        self.layers = torch.nn.ModuleList()
        in_channels += 4
        for _ in range(num_layers):
            mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
            self.layers.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        self.relu = ReLU()

    def forward(self, data):
        x, _ = self.encemb(data)
        edge_index = data.edge_index
        for layer in self.layers:
            x = self.relu(layer(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x)

class BrainNetGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5):
        super(BrainNetGCN, self).__init__()
        self.encemb = BrainEncodeEmbed(functional_groups, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(-1, hidden_channels))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, _ = self.encemb(data)
        edge_index = data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x
    


class BaselineGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, dropout=0.5):
        super(BaselineGCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(-1, hidden_channels))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x

def get_model(args):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    in_channels = args.in_channels
    out_channels = args.num_classes #
    dropout = args.dropout
    groups = yeo_network()
    if model_name == 'gin':
        return BrainNetGIN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'gin_baseline':
        return BaselineGIN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout)
    elif model_name == 'gcn':
        return BrainNetGCN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'gcn_baseline':
        return BaselineGCN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout) 
    else:
        raise ValueError(f'Unknown model name: {model_name}')