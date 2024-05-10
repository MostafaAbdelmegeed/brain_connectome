import graphIO
import sys
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python zero_nans_curv_matrix.py <matrix_filepath>')
        sys.exit(1)
    file_path = sys.argv[1]
    adj_matrix = graphIO.read_adj_matrix_from_file(file_path)
    filename = os.path.basename(file_path)[:-4]
    # Write the matrix to a file, space-delimited
    np.savetxt(f'{filename}_formatted.txt',
               adj_matrix, fmt='%f', delimiter=' ')
    print(f'Matrix written to {filename}_formatted.txt')
