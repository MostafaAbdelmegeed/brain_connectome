import graphIO
import sys
import numpy as np
import pandas as pd
import os
import prep_attr


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python proc_data.py <matrices_dir> <mappings_path>')
        sys.exit(1)
    directory = sys.argv[1]
    mappings_path = sys.argv[2]
    mappings = graphIO.read_mappings_from_json(mappings_path)
    matrices = graphIO.read_adj_matrices_from_directory(directory)
    for filename, matrix in matrices.items():
        # Write the matrix to a file, space-delimited
        np.savetxt(f'{directory}/{filename[:-4]}_proc.txt',
                   matrix, fmt='%f', delimiter=' ')
        prep_attr.prep_attr(matrix, mappings).to_csv(
            f'{directory}/{filename[:-4]}_attributes.csv', index=False)
        print(
            f'Matrix written to {directory}/{filename[:-4]}_proc.txt, and its attributes were written to {directory}/{filename[:-4]}_attributes.csv')
