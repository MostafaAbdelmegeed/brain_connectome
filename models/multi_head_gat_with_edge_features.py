import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.a = nn.Parameter(torch.zeros(size=(heads, 2 * out_features + edge_features, 1)))

        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, edge_attr):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, heads * out_features)
        Wh = Wh.view(-1, self.heads, self.out_features)  # Wh.shape: (N, heads, out_features)
        
        a_input = self._prepare_attentional_mechanism_input(Wh, edge_attr)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # e.shape: (N, N, heads)
        
        attention = torch.where(adj > 0, e, torch.zeros_like(e))
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)  # h_prime.shape: (N, heads, out_features)
        
        if self.concat:
            h_prime = h_prime.view(-1, self.heads * self.out_features)  # h_prime.shape: (N, heads * out_features)
            return F.elu(h_prime)
        else:
            h_prime = h_prime.mean(dim=1)  # h_prime.shape: (N, out_features)
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh, edge_attr):
        N = Wh.size()[0]  # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1, 1)
        edge_attr_repeated = edge_attr.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating, edge_attr_repeated], dim=2)
        return all_combinations_matrix.view(N, N, self.heads, 2 * self.out_features + self.edge_features)

# Example usage
class GATWithEdgeFeatures(nn.Module):
    def __init__(self, in_features, out_features, edge_features, heads, dropout, alpha):
        super(GATWithEdgeFeatures, self).__init__()
        self.layer1 = MultiHeadGATLayerWithEdgeFeatures(in_features, out_features, edge_features, heads, dropout, alpha, concat=True)
        self.layer2 = MultiHeadGATLayerWithEdgeFeatures(out_features * heads, out_features, edge_features, 1, dropout, alpha, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        adj = torch.zeros(x.size(0), x.size(0), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1
        
        x = self.layer1(x, adj, edge_attr)
        x = self.layer2(x, adj, edge_attr)
        return F.log_softmax(x, dim=1)

# Example of creating and using the model
model = GATWithEdgeFeatures(in_features=10, out_features=8, edge_features=4, heads=8, dropout=0.6, alpha=0.2)
