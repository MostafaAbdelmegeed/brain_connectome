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

    def forward(self, h, edge_index, edge_attr):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, heads * out_features)
        Wh = Wh.view(-1, self.heads, self.out_features)  # Wh.shape: (N, heads, out_features)
        
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

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Perform global mean pooling
        x = global_mean_pool(x, batch)
        
        # Classify
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
