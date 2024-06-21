import torch
import numpy as np
import os
from tqdm import tqdm
import torch
from scipy.io import loadmat
from .preprocess import center_matrices

def load_data(dataset_path):
    ppmi_dataset = torch.load(dataset_path)
    connectivity_matrices = center_matrices(ppmi_dataset['data'].numpy())
    connectivity_labels = ppmi_dataset['class_label'].numpy().reshape(-1, 1)
    return connectivity_matrices, connectivity_labels




def write_adj_matrix(adj_matrix_file_path, adj_matrix):
    """
    Writes an adjacency matrix to a file.

    :param adj_matrix_file_path: Path to the file where the adjacency matrix will be written.
    :param adj_matrix: The adjacency matrix to write to the file.
    """
    np.savetxt(adj_matrix_file_path, adj_matrix, fmt='%f')

def read_adj_matrix_from_file(file_path, key='data', as_tensor=False):
    """
    Reads the adjacency matrix from a single file.

    :param file_path: Path to the file containing the adjacency matrix.
    :param as_tensor: Boolean flag to return the matrix as a tensor if True.
    :return: The adjacency matrix as a NumPy array or tensor.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    if os.path.splitext(file_path)[1] == '.mat':
        adj_matrix = np.nan_to_num(loadmat(file_path)[key])
    else:
        adj_matrix = np.nan_to_num(np.loadtxt(file_path))
    if as_tensor:
        adj_matrix = torch.tensor(adj_matrix)
    return adj_matrix

def read_adj_matrices_from_directory(directory, include_string='', as_tensor=False):
    """
    Reads adjacency matrices from files in a directory and stores them in a dictionary.
    The processing of each file is shown with a progress bar.

    :param directory: Path to the directory containing the files.
    :param include_string: String to filter the files to include.
    :param as_tensor: Boolean flag to return matrices as tensors if True.
    :return: A dictionary where the key is the file name and the value is the adjacency matrix.
    """
    adj_matrices = {}  # Dictionary to store adjacency matrices
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Iterate over files in the directory with a progress bar
    for filename in tqdm(files, desc="Reading adjacency matrices"):
        filepath = os.path.join(directory, filename)
        if not include_string or include_string in filename:
            # Read the adjacency matrix from the file
            adj_matrix = read_adj_matrix_from_file(filepath, as_tensor=as_tensor)
            # Store the adjacency matrix in the dictionary
            adj_matrices[os.path.splitext(filename)[0]] = adj_matrix
    return adj_matrices

def parse_class_name(directory_name):
    """
    Parses the class name from the directory name by extracting characters until the first number.
    
    :param directory_name: The name of the directory.
    :return: The class name.
    """
    class_name = ''
    for char in directory_name:
        if char.isdigit():
            break
        class_name += char
    return class_name

def parse_record_name(file_name):
    """
    Parses the record name from the file name by extracting characters before the first underscore.
    
    :param file_name: The name of the file.
    :return: The record name.
    """
    return file_name.split('_')[0]


def read_ppmi_data(ppmi_directory, method='new', as_tensor=False):
    """
    Reads adjacency matrices from files in a directory structure and organizes them in a hierarchical dictionary.
    
    :param parent_directory: Path to the parent directory containing the subdirectories.
    :param as_tensor: Boolean flag to return matrices as tensors if True.
    :return: A dictionary with class names, record names, and their corresponding adjacency matrices.
    """
    hierarchical_dict = {}
    # Get a list of subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(ppmi_directory) if os.path.isdir(os.path.join(ppmi_directory, d))]
    # Iterate over each subdirectory
    for subdirectory in tqdm(subdirectories, desc="Processing directories"):
        class_name = parse_class_name(subdirectory)
        subdirectory_path = os.path.join(ppmi_directory, subdirectory)
        # Get a list of files in the subdirectory
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        for file_name in files:
            if f'_{method}_' in file_name:
                record_name = parse_record_name(file_name)
                file_path = os.path.join(subdirectory_path, file_name)
                # Read the adjacency matrix from the file
                adj_matrix = read_adj_matrix_from_file(file_path, as_tensor=as_tensor)
                # Initialize the class dictionary if it does not exist
                if class_name not in hierarchical_dict:
                    hierarchical_dict[class_name] = {}
                # Add the record and its adjacency matrix to the class dictionary
                hierarchical_dict[class_name][record_name] = adj_matrix
    return hierarchical_dict


def read_ppmi_data_as_tensors(ppmi_directory, atlas='AAL116', method='new'):
    """
    Reads adjacency matrices from files in a directory structure and organizes them in a hierarchical dictionary.
    The adjacency matrices are returned as PyTorch tensors.
    
    :param ppmi_directory: Path to the parent directory containing the subdirectories.
    :return: A dictionary with class names, record names, and their corresponding adjacency matrices as tensors.
    """
    classname_to_label = {'control': 0, 'prodromal': 1, 'patient': 2, 'swedd': 3}
    subdirectories = [d for d in os.listdir(ppmi_directory) if os.path.isdir(os.path.join(ppmi_directory, d))]
    # Iterate over each subdirectory
    data = []
    class_labels = []
    id_labels = []
    for subdirectory in tqdm(subdirectories, desc="Processing directories"):
        class_name = parse_class_name(subdirectory)
        class_label = classname_to_label[class_name[4:]]
        subdirectory_path = os.path.join(ppmi_directory, subdirectory)
        # Get a list of files in the subdirectory
        files = [f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))]
        for file_name in files:
            if f'_{method}_' in file_name and f'_{atlas}_' in file_name:
                record_name = parse_record_name(file_name)
                record_id = int(''.join(filter(str.isdigit, record_name)))
                file_path = os.path.join(subdirectory_path, file_name)
                # Read the adjacency matrix from the file
                adj_matrix = read_adj_matrix_from_file(file_path, as_tensor=True)
                # Initialize the class dictionary if it does not exist
                data.append(adj_matrix)
                class_labels.append(class_label)
                id_labels.append(record_id)
    return {'matrix': torch.stack(data), 'label': torch.tensor(class_labels, dtype=torch.int64), 'id': torch.tensor(id_labels, dtype=torch.int64)}
    


def read_ad_curv_data(ad_directory, as_tensor=False):
    control_matrices = read_adj_matrices_from_directory(f'{ad_directory}/CN-50/new/', include_string='', as_tensor=as_tensor)
    ad_matrices = read_adj_matrices_from_directory(f'{ad_directory}/AD-50/new/', include_string='', as_tensor=as_tensor)
    return center_matrices(np.array(list(control_matrices.values()))), center_matrices(np.array(list(ad_matrices.values())))

def read_ad_adj_data(ad_directory, as_tensor=False):
    control_matrices = read_adj_matrices_from_directory(f'{ad_directory}/CN-50/', include_string='', as_tensor=as_tensor)
    ad_matrices = read_adj_matrices_from_directory(f'{ad_directory}/AD-50/', include_string='', as_tensor=as_tensor)
    return center_matrices(np.array(list(control_matrices.values()))), center_matrices(np.array(list(ad_matrices.values())))


def center_matrices(matrices):
    return standardize_matrices(normalize_matrices(matrices))

def normalize_matrices(matrices):
    """
    Normalize a set of matrices to the range [0, 1].
    
    Parameters:
    matrices (np.ndarray): A numpy array of shape (n, x, x) where n is the number of samples
                           and x is the dimension length of the square matrices.

    Returns:
    np.ndarray: Normalized matrices.
    """
    min_val = np.min(matrices)
    max_val = np.max(matrices)
    normalized = (matrices - min_val) / (max_val - min_val)
    return normalized

def standardize_matrices(matrices):
    """
    Standardize a set of matrices to have zero mean and unit variance.
    
    Parameters:
    matrices (np.ndarray): A numpy array of shape (n, x, x) where n is the number of samples
                           and x is the dimension length of the square matrices.

    Returns:
    np.ndarray: Standardized matrices.
    """
    mean_val = np.mean(matrices)
    std_dev = np.std(matrices)
    standardized = (matrices - mean_val) / std_dev
    return standardized

def analyze_matrices(matrices):
    """
    Analyzes a set of square matrices and prints useful statistics for the entire set.

    Parameters:
    matrices (np.ndarray): A numpy array of shape (n, x, x) where n is the number of samples
                           and x is the dimension length of the square matrices.

    Returns:
    None
    """
    n, x, _ = matrices.shape
    
    # Flatten the matrices to a single array
    all_values = matrices.flatten()
    
    # Calculate statistics
    mean_value = np.mean(all_values)
    std_dev = np.std(all_values)
    max_value = np.max(all_values)
    min_value = np.min(all_values)
    
    # Print the results
    print("Statistics for the entire set of matrices:")
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Maximum Value: {max_value}")
    print(f"Minimum Value: {min_value}")
    print("-" * 40)


def is_symmetric(matrix, tol=1e-8):
    return np.allclose(matrix, matrix.T, atol=tol)




