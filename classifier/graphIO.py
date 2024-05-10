import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm


def read_mappings_from_json(json_file_path):
    """
    Reads mappings from a JSON file and returns them as a dictionary.

    :param json_file_path: Path to the JSON file containing the mappings.
    :return: A dictionary with the mappings read from the JSON file.
    """
    with open(json_file_path, 'r') as file:
        mappings = json.load(file)

    # Convert the keys to integers
    mappings = {int(key.split(' ')[-1]): value for key, value in mappings.items()}

    return mappings

def read_mappings_from_node_file(node_file_path):
    """
    Reads mappings from a node file and returns them as a dictionary.

    :param node_file_path: Path to the node file containing the mappings.
    :return: A dictionary with the mappings read from the node file.
    """
    mappings = {}
    with open(node_file_path, 'r') as file:
        node_id = 1
        for line in file:
            # Split the line by spaces
            parts = line.strip().split(' ')
            # Store the attributes in the dictionary
            mappings[node_id] = parts[-1]
            node_id += 1
    return mappings


def read_nodes(node_file_path):
    """
    Reads nodes from a file and returns them as a list.

    :param node_file_path: Path to the file containing the nodes.
    :return: A list with the nodes read from the file.
    """
    with open(node_file_path, 'r') as file:
        nodes = []
        for line in file:
            # Split each line by space
            split_line = line.strip().split(' ')
            # Convert the split line to appropriate types
            node = [float(split_line[0]), float(split_line[1]), float(split_line[2]), int(split_line[3]), int(split_line[4]), split_line[5]]
            nodes.append(node)
        return nodes
    


def read_attributes(attributes_file_path):
    """
    Reads attributes from a file and returns them as a dictionary.

    :param attributes_file_path: Path to the file containing the attributes.
    :return: A dictionary with the attributes read from the file.
    """
    attributes = pd.read_csv(attributes_file_path, index_col=False)
    return attributes.to_dict(orient='list')
    
def write_attributes(attributes_file_path, attributes):
    df = pd.DataFrame(attributes)
    df.to_csv(attributes_file_path, index=False)

def write_adj_matrix(adj_matrix_file_path, adj_matrix):
    """
    Writes an adjacency matrix to a file.

    :param adj_matrix_file_path: Path to the file where the adjacency matrix will be written.
    :param adj_matrix: The adjacency matrix to write to the file.
    """
    np.savetxt(adj_matrix_file_path, adj_matrix, fmt='%f')


# def convert_to_data_object(graph, label):
#     # Assuming 'graph' is your custom graph object or representation
#     # You need to extract 'edge_index' and 'x' (node features) from your graph

#     # Example conversion (you'll need to adapt this to your graph format)
#     edge_index = ...  # Convert your graph's connectivity to a COO format tensor
#     x = ...  # Extract or define node features as a tensor

#     # Create the Data object with an additional 'y' field for the label
#     data_object = Data(x=x, edge_index=edge_index.t(
#     ).contiguous(), y=torch.tensor([label]))

#     return data_object


def read_adj_matrix_from_file(file_path):
    """
    Reads the adjacency matrix from a single file.

    :param file_path: Path to the file containing the adjacency matrix.
    :return: The adjacency matrix as a NumPy array.
    """

    # adj_matrix = []
    # with open(file_path, 'r') as file:
    #     for line in file:
    #         # Split each line by double spaces and convert the values to floats
    #         row = [float(value) for value in line.strip().split('  ')]
    #         adj_matrix.append(row)
    # return np.array(adj_matrix)
    return np.nan_to_num(np.loadtxt(file_path))


def read_adj_matrices_from_directory(directory):
    """
    Reads adjacency matrices from files in a directory and stores them in a dictionary.
    The processing of each file is shown with a progress bar.

    :param directory: Path to the directory containing the files.
    :return: A dictionary where the key is the file name and the value is the adjacency matrix.
    """
    adj_matrices = {}  # Dictionary to store adjacency matrices
    # Get a list of files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    # Iterate over files in the directory with a progress bar
    for filename in tqdm(files, desc="Reading adjacency matrices"):
        filepath = os.path.join(directory, filename)
        # Read the adjacency matrix from the file
        adj_matrix = read_adj_matrix_from_file(filepath)
        # Store the adjacency matrix in the dictionary
        adj_matrices[os.path.splitext(filename)[0]] = adj_matrix
    return adj_matrices


def read_sp_adj_matrices_from_directory(directory, include_string=''):
    """
    Reads adjacency matrices from files in a directory that include a certain string in their filenames 
    and stores them in a dictionary.

    :param directory: Path to the directory containing the files.
    :param include_string: The string that must be included in the file names.
    :return: A dictionary where the key is the file name and the value is the adjacency matrix.
    """
    adj_matrices = {}  # Dictionary to store adjacency matrices

    # Iterate over files in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)

        # Check if the file is a text file and includes the specified string in its name
        if os.path.isfile(filepath) and include_string in filename:
            # Read the adjacency matrix from the file
            adj_matrix = read_adj_matrix_from_file(filepath)
            # Store the adjacency matrix in the dictionary
            adj_matrices[filename] = adj_matrix

    return adj_matrices


# def read_adj_matrix_and_create_graph(file_path):
#     # Initialize an empty list to store the adjacency matrix
#     adj_matrix_list = []
#     # Read the file
#     with open(file_path, 'r') as file:
#         for line in file:
#             # Split each line by two or more spaces
#             split_line = re.split(r'\s{2,}', line.strip())
#             # Convert the split line to integers and append to the adj_matrix_list
#             adj_matrix_list.append([int(value) for value in split_line])
#     # Convert the list of lists into a NumPy array
#     adj_matrix = np.array(adj_matrix_list)
#     # Initialize an empty graph
#     graph = nx.Graph()
#     # Iterate over the adjacency matrix and add edges only for non-zero values
#     for i in range(adj_matrix.shape[0]):  # Loop over rows
#         for j in range(adj_matrix.shape[1]):  # Loop over columns
#             if adj_matrix[i, j] != 0:
#                 # Add an edge between nodes i and j with weight equal to adj_matrix[i, j]
#                 graph.add_edge(i, j, weight=adj_matrix[i, j])
#     return graph


# def read_adj_matrices_from_directory(directory_path):
#     # Dictionary to store the graphs
#     graphs = {}
#     # Iterate over all files in the directory
#     for filename in os.listdir(directory_path):
#         # Construct the full file path
#         file_path = os.path.join(directory_path, filename)
#         # Check if it's a file
#         if os.path.isfile(file_path):
#             # Use the filename without the extension as the key
#             base_name = os.path.splitext(filename)[0]
#             # Read the graph from the file
#             graph = read_adj_matrix_and_create_graph(file_path)
#             # Add the graph to the dictionary
#             graphs[base_name] = graph
#     return graphs


# def read_adj_matrices_from_directory(directory):
#     adj_matrices = {}  # Dictionary to store adjacency matrices
#     # Iterate over files in the directory
#     for filename in os.listdir(directory):
#         filepath = os.path.join(directory, filename)

#         # Check if the file is a text file (you can adjust the condition based on your file format)
#         if os.path.isfile(filepath) and filename.endswith('.txt'):
#             # Read the adjacency matrix from the file
#             with open(filepath, 'r') as file:
#                 lines = file.readlines()
#                 # Initialize an empty list to store the adjacency matrix
#                 adj_matrix = []
#                 for line in lines:
#                     # Split each line by double spaces and convert the values to integers
#                     row = [int(value) for value in line.strip().split('  ')]
#                     adj_matrix.append(row)
#                 # Store the adjacency matrix in the dictionary
#                 adj_matrices[filename] = adj_matrix

#     return adj_matrices


# def read_nodes_from_excel(node_file, columns=[], ids_col='', ids=[]):
#     """
#     Reads specified columns and rows from an Excel file.

#     :param node_file: Path to the Excel file.
#     :param columns: List of columns to include in the output DataFrame.
#     :param ids_col: Column name that contains the IDs.
#     :param ids: List of IDs to include in the output DataFrame.
#     :return: A pandas DataFrame containing specified columns and rows.
#     """
#     # Read the entire Excel file
#     df = pd.read_excel(node_file, engine='openpyxl')

#     # Filter columns if specified
#     if columns:
#         df = df[columns + [ids_col]] if ids_col not in columns else df[columns]

#     # Filter rows based on ids if specified
#     if ids:
#         df = df[df[ids_col].isin(ids)]

#     return df
