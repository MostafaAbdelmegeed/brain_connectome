import networkx as nx
import numpy as np
import os
import time
from lower_brain import LowerORicci
from new_brain import NewBRicci
from scipy.io import loadmat
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import torch


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

def process_matrix(matrix):
    # Load the .mat file
    A = matrix
    W = (A + np.transpose(A))/2
    return new_bcurv(W)

def main():
    start_time = time.time()

    # Collect all .mat files to be processed
    data = torch.load('data/adni.pth')
    matrices = data['matrix'].numpy()

    # Use ProcessPoolExecutor to process files in parallel
    with ProcessPoolExecutor() as executor:
        results = np.array(list(tqdm(executor.map(process_matrix, matrices), total=len(matrices), desc='Processing Files')))

    torch.save({'matrix': torch.tensor(results), 'label': data['label']}, 'data/adni_curv.pth')
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
