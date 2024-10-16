
import torch
from torch.nn import ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GCNConv, Linear, GINConv
from torch_geometric.nn import MLP as pyg_MLP
from torch_geometric.nn.models import GAT, MLP
from encoding import yeo_7_network, yeo_17_network
import torch.nn.functional as F
import datetime


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")



class BrainContext(torch.nn.Module):
    def __init__(self, functional_groups, group_embed_dim=2, hemi_embed_dim=2, hemi_compartments=2):
        super(BrainContext, self).__init__()
        if functional_groups:
            self.functional_groups = functional_groups
            self.n_groups = len(functional_groups)
            self.hemi_compartments = hemi_compartments
            # Define a learnable embedding layer for functional groups
            self.group_embedding = torch.nn.Embedding(self.n_groups, group_embed_dim)
            self.hemisphere_embedding = torch.nn.Embedding(hemi_compartments, hemi_embed_dim)
            self.embedding_dim = group_embed_dim + hemi_embed_dim
        else:
            self.functional_groups = None
            self.embedding_dim = 0
        
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

    def forward(self, x):
        if not self.functional_groups:
            return x
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
        return x

    

class BrainNetGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5):
        super(BrainNetGIN, self).__init__()
        self.encemb = BrainContext(
            functional_groups=functional_groups
        )
        self.layers = torch.nn.ModuleList()
        in_channels += self.encemb.embedding_dim
        for _ in range(num_layers):
            mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels], dropout=dropout)
            self.layers.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        self.relu = ReLU()

    def forward(self, data):
        x = self.encemb(data.x)
        edge_index = data.edge_index
        for layer in self.layers:
            x = self.relu(layer(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x), x
    

class BrainNetGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5):
        super(BrainNetGCN, self).__init__()
        self.encemb = BrainContext(functional_groups)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(-1, hidden_channels))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x = self.encemb(data.x)
        edge_index = data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea
    


class BrainNetGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5):
        super(BrainNetGAT, self).__init__()
        self.encemb = BrainContext(functional_groups)
        self.model = GAT(in_channels+self.encemb.embedding_dim, hidden_channels, num_layers, dropout=dropout)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x = self.encemb(data.x)
        edge_index = data.edge_index
        x= self.model(x, edge_index, batch=data.batch)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea


class BrainNetMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, functional_groups=None, dropout=0.5):
        super(BrainNetMLP, self).__init__()
        self.encemb = BrainContext(functional_groups)
        self.model = MLP(in_channels=in_channels+self.encemb.embedding_dim, hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, data):
        x = self.encemb(data.x)
        _, x = self.model(x, data.batch, return_emb=True)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea
    
    

def get_model(args):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    in_channels = args.in_channels
    out_channels = args.num_classes #
    dropout = args.dropout
    groups = None
    if args.mode == 'func' or args.mode == 'all':
        groups = yeo_7_network() if args.network == 'yeo7' else yeo_17_network()

    if model_name == 'gin':
        return BrainNetGIN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'gcn':
        return BrainNetGCN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'gat':
        return BrainNetGAT(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'mlp':
        return BrainNetMLP(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=out_channels, num_layers=n_layers, dropout=dropout, functional_groups=groups)
    else:
        raise ValueError(f'Unknown model name: {model_name}')