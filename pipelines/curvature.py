import networkx as nx
import math
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class NewBRicci:
    
    def __init__(self, matrix, G: nx.Graph, weight="weight"):
        self.G = G.copy()
        self.weight = weight
        self.nodenum = len(G.nodes)
        self.matrix = matrix

    # Calculating new curvature for edge (i, j)
    def newcurv(self, i, j):
        w_ij = self.G[i][j]["weight"]

        pre_i = list(nx.all_neighbors(self.G, i))
        n_i = len(pre_i)
        aft_j = list(nx.all_neighbors(self.G, j))
        n_j = len(aft_j)
        w_i = 0
        w_j = 0
        for m in pre_i:
            w_i = w_i +  self.G[m][i]["weight"] 
        for n in aft_j:
            w_j = w_j + self.G[j][n]["weight"]

        frac_i = 0
        frac_j = 0
        for m in list(set(pre_i)-set(aft_j)-set([j])):
            frac_i = frac_i +  math.sqrt(w_ij/self.G[m][i]["weight"])   
        for n in list(set(aft_j)-set(pre_i)-set([i])):
            frac_j = frac_j +  math.sqrt(w_ij/self.G[j][n]["weight"])

        common = []
        tri = 0
        if (set(pre_i)&set(aft_j)):
            common = list(set(pre_i)&set(aft_j))
            for l in common:
                tri = tri +1/(w_ij**2+ self.G[l][i]["weight"]**2 + self.G[j][l]["weight"]**2)
        else:
            common = []
            tri = 0

        ric = 2/max(n_i, n_j)*(w_ij**2)*tri + 1/min(n_i, n_j)*(w_ij**2)*tri
        return ric 

    # Calculating the new curvature for whole network data
    def new_whole(self):
        x = self.matrix.shape[0]
        y = self.matrix.shape[1]
        K = np.empty((x, y))
        K[:] = np.nan
        for i in range(x):
            for j in range(y):
                if self.matrix[i][j] != 0:
                    K[i][j] = self.newcurv(i, j)
        return K

def new_bcurv(matrix):
    G = nx.from_numpy_array(matrix, parallel_edges= False)
    nbc = NewBRicci(matrix, G)
    K3 = nbc.new_whole()
    return K3

def process_matrix(matrix):
    # Load the .mat file
    A = matrix
    W = (A + np.transpose(A))/2
    return new_bcurv(W)