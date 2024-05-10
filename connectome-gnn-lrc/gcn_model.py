import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        pred = output.argmax(dim=1)
        accuracy = (pred == data.y).sum().item() / len(data.y)
        print(f'Accuracy: {accuracy}')


# def get_pyg_data_list(node_features, graph_ids, adj_matrices, labels):
# pyg_data_list = []
# for i in range(len(node_features)):
#     graph_data = create_pyg_data(
#         node_features[i], adj_matrices_dict[graph_ids[i]], labels[i])
#     pyg_data_list.append(graph_data)
# return get_pyg_data_list
