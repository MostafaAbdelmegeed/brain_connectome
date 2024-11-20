
import torch
from torch.nn import ReLU, Sequential
from torch_geometric.nn import global_add_pool, GCNConv, Linear, GINConv, GPSConv, ResGatedGraphConv, GINEConv, GATConv, LayerNorm
from torch_geometric.nn import MLP as pyg_MLP
from torch_geometric.nn.models import GAT, MLP
from torch_geometric.nn.pool import SAGPooling
from networks import yeo_7_network, yeo_17_network
import torch.nn.functional as F
import datetime
import torch.nn as nn


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp}\t{message}")


class BrainContext(nn.Module):
    def __init__(
        self, 
        functional_groups=None, 
        group_embed_dim=2, 
        hemi_embed_dim=2, 
        hemi_compartments=2, 
        include_asymmetry=False, 
        include_connection_type=False, 
        connection_embed_dim=2,
        use_edges=False
    ):
        """
        A module that augments node and edge features for brain connectivity graphs.

        Parameters:
        - functional_groups (dict): Maps functional groups to node indices.
        - group_embed_dim (int): Dimension for functional group embeddings.
        - hemi_embed_dim (int): Dimension for hemisphere embeddings.
        - hemi_compartments (int): Number of hemisphere compartments (typically 2 for L and R).
        - include_asymmetry (bool): Whether to include asymmetry features for nodes.
        - include_connection_type (bool): Whether to add connection type embeddings for edges.
        - connection_embed_dim (int): Dimension for connection type embeddings.
        """
        super(BrainContext, self).__init__()

        self.include_asymmetry = include_asymmetry
        self.include_connection_type = include_connection_type
        self.functional_groups = functional_groups
        self.use_edges = use_edges

        # Set initial output dimensions
        self.node_out_dim = 116
        self.edge_out_dim = 1 if use_edges else None

        # Define embeddings
        if functional_groups:
            self.group_embedding = nn.Embedding(len(functional_groups), group_embed_dim)
            self.hemisphere_embedding = nn.Embedding(hemi_compartments, hemi_embed_dim)
            self.node_out_dim += group_embed_dim + hemi_embed_dim
            self.precomputed_node_embeddings = self._precompute_node_embeddings()
        
        if include_asymmetry:
            # Assuming asymmetry adds half the number of features, one for each hemisphere pair
            self.node_out_dim += 58  # Adjusted for each hemispheric feature pair

        if use_edges and include_connection_type:
            self.connection_embedding = nn.Embedding(3, connection_embed_dim)
            self.edge_out_dim += connection_embed_dim

        # Layer normalization
        self.layer_norm = LayerNorm(self.node_out_dim)

    def _precompute_node_embeddings(self):
        """Precompute functional group and hemisphere embeddings for each node."""
        n_nodes = 116

        # Create a mapping from functional group names (strings) to unique integer IDs
        group_name_to_id = {group_name: idx for idx, group_name in enumerate(self.functional_groups.keys())}

        # Hemisphere embeddings based on node indices
        hemisphere_indicator = torch.arange(n_nodes, requires_grad=False) % 2  # 0 for left, 1 for right
        hemisphere_embeddings = self.hemisphere_embedding(hemisphere_indicator)

        # Assign integer group IDs based on the functional group mapping
        group_ids = torch.zeros(n_nodes, dtype=torch.long, requires_grad=False)
        for group_name, nodes in self.functional_groups.items():
            group_id = group_name_to_id[group_name]
            group_ids[nodes] = group_id  # Assign the integer ID to specified nodes

        # Get functional embeddings based on integer group IDs
        functional_embeddings = self.group_embedding(group_ids)

        # Concatenate the embeddings, no grad required for precomputed node embeddings
        return torch.cat([functional_embeddings, hemisphere_embeddings], dim=-1).detach()

    def _compute_asymmetry_features(self, x):
        """Compute asymmetry features if required."""
        left = x[:, ::2]  # Left hemisphere features
        right = x[:, 1::2]  # Right hemisphere features
        base = torch.abs(left) + torch.abs(right)
        asym = torch.abs(left - right)
        return (base * asym).detach()  # Detach since no grad needed for asymmetry features

    def _compute_connection_types(self, edge_index):
        """Compute edge connection types (homotopic, interhemispheric, intrahemispheric)."""
        src, dst = edge_index[0], edge_index[1]
        homotopic_mask = (src % 2 == 0) & (dst == src + 1)
        interhemispheric_mask = ((src % 2) != (dst % 2)) & ~homotopic_mask
        intrahemispheric_mask = ~homotopic_mask & ~interhemispheric_mask

        connection_types = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device, requires_grad=False)
        connection_types[homotopic_mask] = 0
        connection_types[interhemispheric_mask] = 1
        connection_types[intrahemispheric_mask] = 2
        return connection_types  # No grad needed

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        concat_features = [x]  # Start with original node features

        # Add precomputed node embeddings if functional groups are defined
        if self.functional_groups:
            # Ensure `self.precomputed_node_embeddings` is on the same device as `data.batch`
            precomputed_embeddings = self.precomputed_node_embeddings.to(data.batch.device)
            # Expand embeddings to match batch size using `data.batch`
            precomputed_embeddings_expanded = precomputed_embeddings[data.batch]
            concat_features.append(precomputed_embeddings_expanded)

        # Add asymmetry features if enabled
        if self.include_asymmetry:
            asymmetry_features = self._compute_asymmetry_features(x)
            concat_features.append(asymmetry_features)

        # Concatenate all node features and apply layer normalization
        x = torch.cat(concat_features, dim=-1)
        x = self.layer_norm(x, data.batch)

        # Process edge attributes if connection type embeddings are enabled
        if self.use_edges:
            if self.include_connection_type:
                connection_types = self._compute_connection_types(edge_index)
                connection_embeddings = self.connection_embedding(connection_types).to(edge_attr.device)
                edge_attr = torch.cat([edge_attr.view(-1, 1).float(), connection_embeddings], dim=1)
            else:
                edge_attr = edge_attr.view(-1, 1).float()
        else:
            edge_attr = None

        return x, edge_attr


    @property
    def output_dimensions(self):
        """
        Returns a dictionary with the calculated output dimensions.
        
        Returns:
        - dict: {'node_out_dim': int, 'edge_out_dim': int}
        """
        return {
            'node_out_dim': self.node_out_dim,
            'edge_out_dim': self.edge_out_dim
        }


class BrainNetGIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5, include_asymmetry=False, use_edges=False, include_connection_type=False):
        super(BrainNetGIN, self).__init__()
        self.brain_layer = BrainContext(functional_groups=functional_groups, include_asymmetry=include_asymmetry, include_connection_type=include_connection_type, use_edges=use_edges)
        node_in_channels = self.brain_layer.output_dimensions['node_out_dim']
        edge_in_channels = self.brain_layer.output_dimensions['edge_out_dim'] if use_edges else None
        self.use_edges = use_edges
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = pyg_MLP([node_in_channels, hidden_channels, hidden_channels], dropout=dropout)
            if use_edges:
                self.layers.append(GINEConv(nn=mlp, train_eps=False, edge_dim=edge_in_channels))
            else:
                self.layers.append(GINConv(nn=mlp, train_eps=False))
            node_in_channels = hidden_channels
        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_attr = self.brain_layer(data)
        edge_index = data.edge_index
        for layer in self.layers:
            if self.use_edges:
                x = self.relu(layer(x, edge_index, edge_attr=edge_attr))
            else:
                x = self.relu(layer(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x), x
    

class BrainNetMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, functional_groups=None, dropout=0.5, include_asymmetry=False, use_edges=False, include_connection_type=False):
        super(BrainNetMLP, self).__init__()
        self.brain_layer = BrainContext(functional_groups, include_asymmetry=include_asymmetry, include_connection_type=include_connection_type)
        self.model = MLP(in_channels=self.brain_layer.output_dimensions['node_out_dim'], hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_attr = self.brain_layer(data)
        _, x = self.model(x, data.batch, return_emb=True)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea
    

class BrainNetGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, functional_groups=None, dropout=0.5, include_asymmetry=False, use_edges=False, include_connection_type=False):
        super(BrainNetGCN, self).__init__()
        self.brain_layer = BrainContext(functional_groups, include_asymmetry=include_asymmetry, include_connection_type=include_connection_type, use_edges=use_edges)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GCNConv(-1, hidden_channels))
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_attr = self.brain_layer(data)
        edge_index = data.edge_index
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight=edge_attr)
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
        self.brain_layer = BrainContext(functional_groups)
        self.model = GAT(self.brain_layer.output_dimensions['node_out_dim'], hidden_channels, num_layers, dropout=dropout)
        self.lin1 = Linear(hidden_channels, in_channels)
        self.lin2 = Linear(in_channels, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_attr = self.brain_layer(data)
        edge_index = data.edge_index
        x = self.model(x, edge_index, batch=data.batch, edge_attr=edge_attr)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin1(x)
        x = F.relu(x_fea)
        x = self.lin2(x)
        return x, x_fea

    

class BrainNetGPS(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, functional_groups=None, dropout=0.5, heads=4):
        super(BrainNetGPS, self).__init__()
        self.brain_layer = BrainContext(functional_groups)
        self.convs = torch.nn.ModuleList()
        self.lin1 = Linear(self.brain_layer.output_dimensions['node_out_dim'], hidden_channels)
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
            )
            conv = GPSConv(hidden_channels, GINConv(nn), heads=heads)
            self.convs.append(conv)
        self.mlp = Sequential(
            Linear(hidden_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, out_channels),
        )
        self.lin2 = Linear(hidden_channels, in_channels)
        self.lin3 = Linear(in_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, data):
        x, _ = self.brain_layer(data)
        x = F.relu(self.lin1(x))
        for layer in self.convs:
            x = layer(x, data.edge_index, batch=data.batch)
        x = global_add_pool(x, data.batch) 
        x_fea = self.lin2(x)
        x = F.relu(x_fea)
        x = self.lin3(x)
        return x, x_fea
    

    
    

def get_model(args):
    model_name = args.model
    hidden_dim = args.hidden_dim
    n_layers = args.n_layers
    in_channels = 116
    out_channels = args.num_classes
    dropout = args.dropout
    include_asymmetry = args.include_asymmetry
    use_edges = args.use_edges
    include_connection_type = args.include_connection_type
    groups = None
    if args.mode == 'function':
        groups = yeo_7_network() if args.network == 'yeo7' else yeo_17_network()
    if model_name == 'gin':
        return BrainNetGIN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, 
                           dropout=dropout, functional_groups=groups, include_asymmetry=include_asymmetry, use_edges=use_edges, 
                           include_connection_type=include_connection_type)
    elif model_name == 'gcn':
        return BrainNetGCN(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'gat':
        return BrainNetGAT(in_channels=in_channels, hidden_channels=hidden_dim, num_layers=n_layers, out_channels=out_channels, dropout=dropout, functional_groups=groups)
    elif model_name == 'mlp':
        return BrainNetMLP(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=out_channels, num_layers=n_layers, dropout=dropout, functional_groups=groups)
    elif model_name == 'gps':
        return BrainNetGPS(in_channels=in_channels, hidden_channels=hidden_dim, out_channels=out_channels, num_layers=n_layers, dropout=dropout, functional_groups=groups, heads=args.heads)
    else:
        raise ValueError(f'Unknown model name: {model_name}')