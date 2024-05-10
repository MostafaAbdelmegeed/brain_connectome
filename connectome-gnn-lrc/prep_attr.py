import graphIO
import pandas as pd
import sys
import numpy as np
import os
import helpers


def prep_attr(adj_mat):
    connectivity = helpers.normalize_array(np.count_nonzero(adj_mat, axis=1))
    strength = helpers.normalize_array(adj_mat.sum(axis=1))
    df = pd.DataFrame({'strength': strength,
                       'connectivity': connectivity})
    return df.to_dict('list')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python prep_attr.py <connectivity_matrix_path> <regions_filepath>')
        sys.exit(1)
    file_path = sys.argv[1]
    file_path_2 = sys.argv[2]
    filename = os.path.basename(file_path)[:-4]
    filename_2 = os.path.basename(file_path_2)[:-4]
    matrix = graphIO.read_adj_matrix_from_file(file_path)
    region_name_mappings = graphIO.read_mappings_from_json(
        file_path_2)
    df = prep_attr(file_path, file_path_2)
    df.to_csv(f'{filename}_attributes.csv', index=False)
