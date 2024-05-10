import networkx as nx
import numpy as np
import os
import time
from lower_brain import LowerORicci
from new_brain import NewBRicci
from scipy.io import loadmat
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

base_path = 'C:/Users/mosta\OneDrive - UNCG/Academics/CSC 699 - Thesis/data/ppmi'


def lowercurv(matrix):
    G = nx.from_numpy_array(matrix, parallel_edges= False)
    lrc = LowerORicci(matrix, G)
    K2 = lrc.lower_curvature2()
    return K2

def new_bcurv(matrix):
    G = nx.from_numpy_array(matrix, parallel_edges= False)
    nbc = NewBRicci(matrix, G)
    K3 = nbc.new_whole()
    return K3

def process_file(file_path):
    # Load the .mat file
    mat_contents = loadmat(file_path)
    A = mat_contents['data']
    W = (A + np.transpose(A))/2

    # Compute curvatures
    K2 = lowercurv(W)
    K3 = new_bcurv(W)

    # Prepare save paths
    lrc_save_file_path = file_path.replace('_correlation_matrix.mat', '_lrc_matrix.txt')
    new_save_file_path = file_path.replace('_correlation_matrix.mat', '_new_matrix.txt')

    # Save output files
    np.savetxt(lrc_save_file_path, K2, fmt='%10.5f', delimiter=' ')
    np.savetxt(new_save_file_path, K3, fmt='%10.5f', delimiter=' ')

    return file_path  # Return the path of the processed file for progress tracking

def main():
    start_time = time.time()

    # Collect all .mat files to be processed
    mat_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if 'AAL116_correlation_matrix' in file:
                mat_files.append(os.path.join(root, file))

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, mat_files), total=len(mat_files), desc='Processing Files'))

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

if __name__ == '__main__':
    main()

# start_time = time.time()
# # Specify the directory path
# directory_path = '/brain/Data2/AD-50'

# # Loop through each file in the directory
# for filename in os.listdir(directory_path):
#     # Construct the full file path
#     file_path = os.path.join(directory_path, filename)
#     # Check if it is a file (and not a directory)
#     if os.path.isfile(file_path):
#         A = np.loadtxt(file_path)
#         W = (A + np.transpose(A))/2
#         K2 = lowercurv(W)
#         K3 = new_bcurv(W)
#         np.savetxt(f'/brain/Data2/Result/AD-50/{filename}_lrc2.txt',K2, fmt='%10.5f', delimiter=' ')
#         np.savetxt(f'/brain/Data2/Result/AD-50/{filename}_new2.txt',K3, fmt='%10.5f', delimiter=' ')
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution time: {execution_time} seconds")
