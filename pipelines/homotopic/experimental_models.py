import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool, MLP as pyg_MLP
from einops.layers.torch import Rearrange
import numpy as np

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.lin1 = Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.lin2 = Linear(hidden_size, out_feats)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # Add a small epsilon to edge weights to avoid zeros
        edge_weight[:, 0] = edge_weight[:, 0] + 1e-8

        x = self.conv1(x, edge_index, edge_weight[:, 0])
        x = self.ln1(x)
        print("After conv1:", x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight[:, 0])
        x = self.ln2(x)
        print("After conv2:", x)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch)
        x = F.dropout(x, p=0.3, training=self.training)
        x_fea = self.lin1(x)
        x_fea = self.ln3(x_fea)
        print("After lin1:", x_fea)
        x = F.relu(x_fea)
        x = self.lin2(x)
        print("Final output:", x)
        return x, x_fea


# GIN Model
class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = pyg_MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp))
            in_channels = hidden_channels
        self.mlp = pyg_MLP([hidden_channels, hidden_channels, out_channels], dropout=0.5)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_add_pool(x, data.batch)
        return self.mlp(x), x

# MLP Model
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_classes, dropout):
        super(MLP, self).__init__()
        self.layer0 = Sequential(Linear(in_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU())
        self.layer1 = Sequential(Linear(hidden_dim, out_dim), BatchNorm1d(out_dim), ReLU())
        self.drop = nn.Dropout(p=dropout)
        self.classify = nn.Linear(out_dim, num_classes)

    def forward(self, data):
        x = data.x.view(data.num_graphs, -1)  # Flatten the node features
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h = self.drop(h1)
        hg = self.classify(h)
        return hg, h1

# Temporal Convolution for CNN_1D
class Temporal_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation):
        super(Temporal_Conv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)
        self.activation = activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.activation(h)
        h = self.pool(h)
        return h

# CNN_1D Model
class CNN_1D(nn.Module):
    def __init__(self, nrois, f1, f2, dilation_exponential, k1, dropout, readout, num_classes):
        super(CNN_1D, self).__init__()
        self.readout = readout
        self.layer0 = Temporal_Conv(nrois, f1, k1, dilation_exponential, F.relu)
        self.layer1 = Temporal_Conv(f1, f2, k1, dilation_exponential, F.relu)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        dim = 2 if readout == 'meanmax' else 1
        self.drop = nn.Dropout(p=dropout)
        self.classify = nn.Linear(f2 * dim, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = self.layer0(x)
        h1 = self.layer1(h0)
        h_avg = torch.squeeze(self.avg(h1))
        h_max = torch.squeeze(self.max(h1))
        if self.readout == 'meanmax':
            h = torch.cat((h_avg, h_max), 1)
        else:
            h = h_avg
        h = self.drop(h)
        hg = self.classify(h)
        return hg, h

# LSTM Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        x = self.fc1(out[:, -1, :])
        pre = F.relu(x)
        pre = self.fc(pre)
        return pre, x

# RNN Classifier
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        x = self.fc1(out[:, -1, :])
        pre = F.relu(x)
        pre = self.fc(pre)
        return pre, x

# MLP Mixer
class Mixer_Embedding(nn.Module):
    def __init__(self, input_dim):
        super(Mixer_Embedding, self).__init__()
        self.tok_embed = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.tok_embed(x)

class MixerBlock(nn.Module):
    def __init__(self, input_dim, length, tokens_mlp_dim, channels_mlp_dim):
        super(MixerBlock, self).__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            Rearrange('b n d -> b d n'),
            nn.Linear(length, tokens_mlp_dim),
            nn.ReLU(),
            nn.Linear(tokens_mlp_dim, length),
            Rearrange('b d n -> b n d')
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, channels_mlp_dim),
            nn.ReLU(),
            nn.Linear(channels_mlp_dim, input_dim)
        )

    def forward(self, x):
        x = x + self.token_mixing(x)
        x = x + self.channel_mixing(x)
        return x

class MLP_Mixer(nn.Module):
    def __init__(self, input_dim, length, tokens_mlp_dim, channels_mlp_dim, num_classes, num_blocks):
        super(MLP_Mixer, self).__init__()
        self.embedding = Mixer_Embedding(input_dim)
        self.input_dim = input_dim
        self.num_blocks = num_blocks
        self.mixer_blocks = nn.ModuleList([MixerBlock(input_dim, length, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)])
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, data):
        x = self.embedding(data.x)
        for block in self.mixer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out, x

# Transformer (TF) Model
class Embedding(nn.Module):
    def __init__(self, in_features, d_model, length, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Linear(in_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, length + 1, d_model))
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, self.tok_embed(x)), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        return self.dropout(x)

class TF(nn.Module):
    def __init__(self, in_features, d_model, n_heads, d_ff, length, num_layers, num_classes, device):
        super(TF, self).__init__()
        self.embedding = Embedding(in_features, d_model, length, device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.2,
            activation='relu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc1 = nn.Linear(d_model, in_features)
        self.cls_head = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc1(x[:, 0])
        pred_clf = self.cls_head(x)
        return pred_clf, x
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

class MultiHeadGATLayerWithEdgeFeatures(nn.Module):
    def __init__(self, in_features, out_features, edge_features, heads, dropout, alpha, concat=True):
        super(MultiHeadGATLayerWithEdgeFeatures, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, heads * out_features)))
        self.a = nn.Parameter(torch.zeros(size=(heads, 2 * out_features + edge_features)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, edge_index, edge_attr, edge_type_filter=None):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, heads * out_features)
        Wh = Wh.view(-1, self.heads, self.out_features)  # Wh.shape: (N, heads, out_features)

        if edge_type_filter is not None:
            # Filter edges based on edge_type_filter
            mask = (edge_attr[:, edge_type_filter] > 0)
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]
        
        a_input = self._prepare_attentional_mechanism_input(Wh, edge_index, edge_attr)

        e = self.leakyrelu(torch.einsum('ehd,hd->eh', a_input, self.a))  # e.shape: (E, heads)
        
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.zeros_like(Wh)
        for i in range(edge_index.size(1)):
            h_prime[edge_index[0, i]] += attention[i].unsqueeze(-1) * Wh[edge_index[1, i]]
        
        if self.concat:
            h_prime = h_prime.view(-1, self.heads * self.out_features)  # h_prime.shape: (N, heads * out_features)
            return F.elu(h_prime)
        else:
            h_prime = h_prime.mean(dim=1)  # h_prime.shape: (N, out_features)
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, edge_index, edge_attr):
        Wh_i = Wh[edge_index[0]]  # (E, heads, out_features)
        Wh_j = Wh[edge_index[1]]  # (E, heads, out_features)
        
        # Expand edge_attr to match the dimensions (E, heads, edge_features)
        edge_attr_expanded = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E, heads, edge_features)
        
        all_combinations_matrix = torch.cat([Wh_i, Wh_j, edge_attr_expanded], dim=2)  # (E, heads, 2 * out_features + edge_features)
        return all_combinations_matrix

class GATWithEdgeFeatures(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, edge_features, heads, dropout, alpha, num_classes, num_layers):
        super(GATWithEdgeFeatures, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        self.layers.append(MultiHeadGATLayerWithEdgeFeatures(in_features, hidden_features, edge_features, heads, dropout, alpha, concat=True))
        for _ in range(num_layers - 2):
            self.layers.append(MultiHeadGATLayerWithEdgeFeatures(hidden_features * heads, hidden_features, edge_features, heads, dropout, alpha, concat=True))
        self.layers.append(MultiHeadGATLayerWithEdgeFeatures(hidden_features * heads, out_features, edge_features, 1, dropout, alpha, concat=False))
        
        self.classifier = nn.Linear(out_features, num_classes)

    def forward(self, data, edge_type_filter=None):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, edge_type_filter=edge_type_filter)
        
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        
        # Classify
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    

def get_model(args, device, node_features, hidden, out_features, layers, n_classes, connectivity_matrices):
    if args.model == 'GCN':
        model = GCN(in_feats=node_features, hidden_size=hidden, out_feats=out_features).to(device)
    elif args.model == 'GIN':
        model = GIN(in_channels=node_features, hidden_channels=hidden, out_channels=out_features, num_layers=layers).to(device)
    elif args.model == 'MLP':
        model = MLP(in_dim=node_features, hidden_dim=hidden, out_dim=out_features, num_classes=n_classes, dropout=args.drop_out).to(device)
    elif args.model == 'CNN_1D':
        model = CNN_1D(nrois=node_features, f1=hidden, f2=out_features, dilation_exponential=2, k1=3, dropout=args.drop_out, readout='mean', num_classes=n_classes).to(device)
    elif args.model == 'LSTM':
        model = LSTMClassifier(input_size=node_features, hidden_size=hidden, num_layers=layers, num_classes=n_classes, device=device).to(device)
    elif args.model == 'RNN':
        model = RNNClassifier(input_size=node_features, hidden_size=hidden, num_layers=layers, num_classes=n_classes, device=device).to(device)
    elif args.model == 'MLP_Mixer':
        model = MLP_Mixer(input_dim=node_features, length=connectivity_matrices.shape[2], tokens_mlp_dim=256, channels_mlp_dim=hidden, num_classes=n_classes, num_blocks=4).to(device)
    elif args.model == 'TF':
        model = TF(in_features=node_features, d_model=hidden, n_heads=2, d_ff=hidden * 4, length=connectivity_matrices.shape[2], num_layers=layers, num_classes=n_classes, device=device).to(device)
    elif args.model == 'GATEdge':
        model = GATWithEdgeFeatures(in_features=node_features, hidden_features=hidden, out_features=out_features, edge_features=6, heads=2, dropout=0.6, alpha=0.2, num_classes=n_classes, num_layers=layers).to(device)
    else:
        raise ValueError("Unsupported model type")
    return model


