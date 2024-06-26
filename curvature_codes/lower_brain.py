import networkx as nx
import math
import numpy as np

                
class LowerORicci:
    
    def __init__(self, matrix, G: nx.Graph, weight="weight"):
        self.G = G.copy()
        self.weight = weight
        self.nodenum = len(G.nodes)
        self.matrix = matrix
    
    # Returns in matrix form
    def lower_curvature2(self):
        x = self.matrix.shape[0]
        y = self.matrix.shape[1]
        C = np.empty((x, y))
        C[:] = np.nan
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            n_max = max(n_i, n_j)
            n_min = min(n_i, n_j)
            C[i][j] = 2/n_i + 2/n_j - 2 + 2*n_ij/n_max + n_ij/n_min
            C[j][i] = 2/n_i + 2/n_j - 2 + 2*n_ij/n_max + n_ij/n_min
        
        return C
    
    # Returns in networkx graph form 
    def lower_curvature(self):
        C = {}
        for i, j in self.G.edges:
            n_ij = len(sorted(nx.common_neighbors(self.G, i, j)))
            n_i = len(sorted(nx.all_neighbors(self.G, i)))
            n_j = len(sorted(nx.all_neighbors(self.G, j)))
            n_max = max(n_i, n_j)
            n_min = min(n_i, n_j)
            C[(i, j)] = 2/n_i + 2/n_j - 2 + 2*n_ij/n_max + n_ij/n_min
        nx.set_edge_attributes(self.G, C, "low")
        return self.G