import torch
import numpy as np
import re
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils.sparse import to_torch_coo_tensor
from torch_geometric.data import Data

def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def preprocess_adjacency_matrix(adjacency_matrix, percent):
    top_percent = np.percentile(adjacency_matrix.flatten(), 100-percent)
    adjacency_matrix[adjacency_matrix < top_percent] = 0
    data_adj = from_scipy_sparse_matrix(sp.coo_matrix(adjacency_matrix))
    return data_adj

def add_structural_features(edge_index, edge_attr):
    """
    This function adds structural edge features:
    - Interhemispheric (1) or Intrahemispheric (0)
    - Homotopic (0 or 1)
    """

    # Initialize lists to hold the new features
    hemispheric_feature = []
    homotopic = []

    # Iterate over edges to determine the type of connection
    for i, j in zip(edge_index[0], edge_index[1]):
        region_i = i.item()  # From node
        region_j = j.item()  # To node

        # Hemispheric feature: Interhemispheric = 1, Intrahemispheric = 0
        if (region_i % 2 != region_j % 2):  # Different hemispheres
            hemispheric_feature.append(1)  # Interhemispheric
        else:
            hemispheric_feature.append(0)  # Intrahemispheric

        # Homotopic: Same region but across hemispheres (e.g., 0 <-> 1, 2 <-> 3)
        if abs(region_i - region_j) == 1 and (region_i // 2 == region_j // 2):
            homotopic.append(1)
        else:
            homotopic.append(0)

    # Add the new features to edge_attr
    edge_attr = torch.cat([
        edge_attr.unsqueeze(1),  # Original edge weight
        torch.tensor(hemispheric_feature).unsqueeze(1).float(),  # Hemispheric feature (0 or 1)
        torch.tensor(homotopic).unsqueeze(1).float()  # Homotopic (0 or 1)
    ], dim=1)

    return edge_attr


def pearson_dataset(data, label, sparsity, mgnn=False):
    adjacency_matrix = preprocess_adjacency_matrix(data, sparsity)
    edge_index = adjacency_matrix[0]
    edge_attr = adjacency_matrix[1]
    if mgnn:
        moment_attrs = add_attributes2(edge_index, 10)
        data = Data(x=torch.cat((data,moment_attrs), dim=1).float(), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        return data
    data = Data(x=data.float(), edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
    return data


def compute_diagonal(S):
    n = S.shape[0]
    diag = torch.zeros(n)
    for i in range(n):
        diag[i] = S[i,i] 
    return diag



''' 
below is copy from original MomentGNN to replicate the model
github: https://github.com/MomentGNN/Counting

Kanatsoulis, C., & Ribeiro, A. (2023, October). 
Counting Graph Substructures with Graph Neural Networks. 
In The Twelfth International Conference on Learning Representations.
'''
def add_attributes2(edges, K, device='cpu'):
    S = to_torch_coo_tensor(edges).to(device)
    N = S.shape[0]
    deg_k = torch.zeros(N,K, device=device)
    diag_k = torch.zeros(N,K-2, device=device)
    x = S
    for k in range(K):
        deg_k[:,k] = torch.sparse.sum(x,1).to_dense()
        # x = torch.sparse.mm(S, x)/math.factorial(k+1)
        if k > 1:
            # diag_k[:,k-1] = compute_diagonal(x)/math.factorial(k+1)
            diag_k[:,k-2] = compute_diagonal(x)
        x = torch.sparse.mm(S, x)

    if K > 9:
        new = torch.zeros(N,27, device=device)
    elif K > 8:
        new = torch.zeros(N,22, device=device)
    elif K > 7:
        new = torch.zeros(N,16, device=device)
    elif K > 5:
        new = torch.zeros(N,7, device=device)

    if K > 5:
        I = torch.eye(N).to_sparse().to(device)
        S2 = torch.sparse.mm(S, S)
        S3 = torch.sparse.mm(S, S2)

        if torch.sparse.sum(I * S2 * S2) != 0:
            new[:,0] = torch.sparse.sum(I * S2 * S2,1).to_dense()
        if torch.sparse.sum(I * S2 * S3) != 0:
            new[:,1] = torch.sparse.sum(I * S2 * S3,1).to_dense()

        if torch.sparse.sum(I * S3 * S3) != 0:
            new[:,2] = torch.sparse.sum(I * S3 * S3,1).to_dense()


        if torch.sparse.sum(S * S2) != 0:
            new[:,3] = torch.sparse.sum(S * S2 * S2,1).to_dense()
        if torch.sparse.sum(S * S2 * S3) != 0:
            new[:,4] = torch.sparse.sum(S * S2 * S3,1).to_dense()

        if torch.sparse.sum(I * S2) != 0:
                    new[:,5] = torch.sparse.sum(I * S2 * S2 * S2,1).to_dense()
        new[:,6] = torch.sparse.sum(S2 * S2 * S2,1).to_dense()


        if K > 7:
            new[:,7] = torch.sparse.sum(S * S3 * S3,1).to_dense()
            if torch.sparse.sum(S2 * S3) != 0:
                new[:,8] = torch.sparse.sum(S2 * S2 * S3,1).to_dense()
                new[:,9] = torch.sparse.sum(S2 * S3 * S3,1).to_dense()
            if torch.sparse.sum(S * S2) != 0:
                new[:,10] = torch.sparse.sum(S * S2 * S2 * S2,1).to_dense()
            if torch.sparse.sum(S * S2 * S3) != 0:
                new[:,11] = torch.sparse.sum(S * S2 * S2 * S3,1).to_dense()

            if torch.sparse.sum(I * S2 * S3) != 0:
                new[:,12] = torch.sparse.sum(I * S2 * S2 * S3,1).to_dense()

            if torch.sparse.sum(I * S2 * S3) != 0:
                new[:,13] = torch.sparse.sum(I * S2 * S3 * S3,1).to_dense()

            if torch.sparse.sum(S2 * I) != 0:
                new[:,14] = torch.sparse.sum(I * S2 * S2 * S2 * S2,1).to_dense()
            new[:,15] = torch.sparse.sum(S2 * S2 * S2 * S2,1).to_dense()

            if K > 8:

                new[:,16] = torch.sparse.sum(S3 * S3 * S3,1).to_dense()
                if torch.sparse.sum(S * S2 * S3) != 0:
                    new[:,17] = torch.sparse.sum(S * S2 * S3 * S3,1).to_dense()
                if torch.sparse.sum(I * S3) != 0:
                    new[:,18] = torch.sparse.sum(I * S3 * S3 * S3,1).to_dense()
                if torch.sparse.sum(S2 * S3) != 0:
                    new[:,19] = torch.sparse.sum(S2 * S2 * S2 * S3,1).to_dense()
                if torch.sparse.sum(I * S2 * S3) != 0:
                    new[:,20] = torch.sparse.sum(I * S3 * S2 * S2 * S2,1).to_dense()
                if torch.sparse.sum(S2 * S) != 0:
                    new[:,21] = torch.sparse.sum(S * S2 * S2 * S2 * S2,1).to_dense()


                if K > 9:
                    if torch.sparse.sum(S * S3) != 0:
                        new[:,22] = torch.sparse.sum(S * S3 * S3 * S3,1).to_dense()

                    if torch.sparse.sum(S2 * S3) != 0:
                        new[:,23] = torch.sparse.sum(S2 * S2 * S3 * S3,1).to_dense()
                    if torch.sparse.sum(I * S2 * S3) != 0:
                        new[:,24] = torch.sparse.sum(I * S3 * S3 * S2 * S2,1).to_dense()
                    if torch.sparse.sum(S * S2 * S3) != 0:
                        new[:,25] = torch.sparse.sum(S * S2 * S2 * S2 * S3,1).to_dense()

                    new[:,26] = torch.sparse.sum(S2 * S2 * S2 * S2 * S2,1).to_dense()
        

    if K > 5:
        y = torch.cat((diag_k, deg_k, new), 1)
        y[y==0] = 0.5
        y = torch.log(y)
    else:
        y = torch.cat((diag_k, deg_k), 1)
        y[y==0] = 0.5
        y = torch.log(y)
    return y





