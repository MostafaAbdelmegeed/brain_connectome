import torch
import numpy as np

# Load data
data = torch.load('data/ppmi.pth')
connectivity = data['matrix']

# Function to calculate the Frobenius norm between two matrices
def frobenius_norm(mat1, mat2):
    return torch.norm(mat1 - mat2, p='fro')

# Calculate pairwise Frobenius norms
num_matrices = connectivity.shape[0]
distances = np.zeros((num_matrices, num_matrices))

for i in range(num_matrices):
    for j in range(i + 1, num_matrices):
        distance = frobenius_norm(connectivity[i], connectivity[j])
        distances[i, j] = distance
        distances[j, i] = distance

# Print summary statistics of the distances
distances_flat = distances[np.triu_indices(num_matrices, k=1)]
print(f'Mean pairwise Frobenius norm: {distances_flat.mean()}')
print(f'Standard deviation of pairwise Frobenius norms: {distances_flat.std()}')
print(f'Minimum pairwise Frobenius norm: {distances_flat.min()}')
print(f'Maximum pairwise Frobenius norm: {distances_flat.max()}')

# # If you want to check the pairwise distances for each matrix
# for i in range(num_matrices):
#     for j in range(i + 1, num_matrices):
#         print(f'Distance between matrix {i} and matrix {j}: {distances[i, j]}')
