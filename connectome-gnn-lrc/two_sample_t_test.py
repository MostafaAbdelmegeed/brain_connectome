import argparse
import graphIO
import numpy as np
from scipy.stats import ttest_ind
import datetime


def two_sample_t_test(arrays_1, arrays_2, significance=1.0, save=False, output_path='./output/', axis=0):
    if axis == 0:
        return two_sample_t_test_arrays(arrays_1, arrays_2, significance, save, output_path)
    elif axis == 1:
        return two_sample_t_test_matrices(arrays_1, arrays_2, significance, save, output_path)


def two_sample_t_test_arrays(arrays_1, arrays_2, significance, save=False, output_path='./output/'):
    n_arrays = len(arrays_1)
    n_features = len(arrays_1[0])
    p_values = np.zeros(n_features)
    for i in range(n_features):
        values_a = np.array([array[i] for array in arrays_1])
        values_b = np.array([array[i] for array in arrays_2])
        _, p_value = ttest_ind(values_a, values_b)
        p_values[i] = p_value

    num_p_values_below_significance = len(
        p_values[p_values < significance])
    p_values[p_values > significance] = np.nan
    print(
        f"The number of p-values below the significance level is: {num_p_values_below_significance}")
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save:
        np.savetxt(f"{output_path}{date_time}.csv",
                   p_values, delimiter=",")
        np.save(f"{output_path}{date_time}.npy", p_values)
    return p_values


def two_sample_t_test_matrices(matrices_1, matrices_2, significance, save=False, output_path='./output/'):
    n_rows, n_cols = matrices_1[0].shape
    p_values_matrix = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            # Collect all elements from the current position across all matrices in each class
            values_a = np.array([matrix[i, j] for matrix in matrices_1])
            values_b = np.array([matrix[i, j] for matrix in matrices_2])

            # Perform the two-sample t-test for the current position
            _, p_value = ttest_ind(values_a, values_b)
            p_values_matrix[i, j] = p_value

    num_p_values_below_significance = len(
        p_values_matrix[p_values_matrix < significance])
    p_values_matrix[p_values_matrix > significance] = np.nan
    print(
        f"The number of p-values below the significance level is: {num_p_values_below_significance}")
    # Get the current date and time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save:
        # Save the p-values matrix as a csv file with the current date and time in the filename
        np.savetxt(f"{output_path}{date_time}.csv",
                   p_values_matrix, delimiter=",")
        # Save the p_values_matrix as a numpy array
        np.save(f"{output_path}{date_time}.npy", p_values_matrix)
    return p_values_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Process two sets of matrices.')
    parser.add_argument('dir1', type=str, help='First directory')
    parser.add_argument('dir2', type=str, help='Second directory')
    parser.add_argument('--method', type=str, choices=[
                        'new', 'lrc'], default='new', help='Choose a method (new or lrc)')
    parser.add_argument('--significance', type=float, default=0.05,
                        help='Significance value (between 0 and 1)')
    parser.add_argument('--save', action='store_true',
                        help='Save p-value matrix as a csv file')
    parser.add_argument('--output_path', type=str, default='./output/',
                        help='Output path for saving the csv file')

    args = parser.parse_args()

    # Access the arguments
    dir1 = args.dir1
    dir2 = args.dir2
    CALC_METHOD = args.method + '_proc'
    SIGNIFICANCE_LEVEL = args.significance
    output_path = args.output_path
    save_flag = args.save

    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print(f"Method: {CALC_METHOD}")
    print(f"Significance: {SIGNIFICANCE_LEVEL}")

    CN_matrices = list(graphIO.read_sp_adj_matrices_from_directory(
        f"{dir1}{CALC_METHOD}/", include_string=CALC_METHOD).values())

    AD_matrices = list(graphIO.read_sp_adj_matrices_from_directory(
        f"{dir2}{CALC_METHOD}/", include_string=CALC_METHOD).values())

    n_rows, n_cols = CN_matrices[0].shape
    p_values_matrix = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        for j in range(n_cols):
            # Collect all elements from the current position across all matrices in each class
            values_a = np.array([matrix[i, j] for matrix in CN_matrices])
            values_b = np.array([matrix[i, j] for matrix in AD_matrices])

            # Perform the two-sample t-test for the current position
            _, p_value = ttest_ind(values_a, values_b)
            p_values_matrix[i, j] = p_value

    num_p_values_below_significance = len(
        p_values_matrix[p_values_matrix < SIGNIFICANCE_LEVEL])
    print(
        f"The number of p-values below the significance level is: {num_p_values_below_significance}")
    # Get the current date and time
    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if save_flag:
        # Save the p-values matrix as a csv file with the current date and time in the filename
        np.savetxt(f"{output_path}{date_time}.csv",
                   p_values_matrix, delimiter=",")
    # Save the p_values_matrix as a numpy array
    np.save(f"{output_path}{CALC_METHOD}_{date_time}.npy", p_values_matrix)


if __name__ == '__main__':
    main()
