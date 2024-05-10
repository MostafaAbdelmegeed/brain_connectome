import numpy as np


def normalize_array(arr):
    """
    Normalize a numpy array values between 0 and 1.

    Parameters:
    - arr: NumPy array to be normalized.

    Returns:
    - Normalized NumPy array with values between 0 and 1.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)

    # Avoid division by zero if the array is constant
    if max_val - min_val == 0:
        return arr - min_val
    else:
        return (arr - min_val) / (max_val - min_val)
    
def normalize_matrix(matrix):
    """
    Normalize a matrix by its maximum value.

    Parameters:
    - matrix: NumPy array representing the matrix.

    Returns:
    - Normalized NumPy array with values between 0 and 1.
    """

    return matrix / np.max(matrix)
    

def save_adj_matrix(matrix, filename):
    """
    Save an adjacency matrix to a text file.

    Parameters:
    - matrix: NumPy array representing the adjacency matrix.
    - filename: Name of the file to save the matrix to.
    """
    np.savetxt(filename, matrix, fmt='%f', delimiter=' ')
