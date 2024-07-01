import numpy as np
from scipy.stats import ttest_ind

###################### AAL116 MATH

# def calculate_interhemispheric(matrix, method='abs_diff', epsilon=1e-10):
#     means = np.mean(matrix, axis=1)
#     ai = []
#     for i in range(0, len(means)-1, 2):
#         if method == 'abs_diff':
#             ai.append(np.abs(means[i] - means[i+1]))
#         elif method == 'ai':
#             ai.append((means[i] - means[i+1]) / (means[i] + means[i+1] + epsilon))
#         elif method == 'sq_diff':
#             ai.append((means[i] - means[i+1])**2)
#         elif method == 'log_ratio':
#             ai.append(np.log((means[i] + epsilon) / (means[i+1] + epsilon)))
#         elif method == 'perc_diff':
#             ai.append((means[i] - means[i+1]) / ((means[i] + means[i+1] + epsilon) / 2) * 100)
#         elif method == 'ratio':
#             ai.append((means[i] + epsilon) / (means[i+1] + epsilon))
#         elif method == 'smape':
#             ai.append(100 * (np.abs(means[i] - means[i+1]) / ((np.abs(means[i]) + np.abs(means[i+1]) + epsilon) / 2)))
#     return ai

def compute_intrahemispherical_asymmetry(matrix, method='absolute', num_regions=116):
    left_hemisphere_indices = np.arange(0, num_regions, 2)
    right_hemisphere_indices = np.arange(1, num_regions, 2)
    if method == 'index':
        asymmetry = (matrix[np.ix_(left_hemisphere_indices, left_hemisphere_indices)] - matrix[np.ix_(right_hemisphere_indices, right_hemisphere_indices)])/(matrix[np.ix_(left_hemisphere_indices, left_hemisphere_indices)] + matrix[np.ix_(right_hemisphere_indices, right_hemisphere_indices)])
    else:
        asymmetry = np.abs(matrix[np.ix_(left_hemisphere_indices, left_hemisphere_indices)] - matrix[np.ix_(right_hemisphere_indices, right_hemisphere_indices)])
    return asymmetry

def compute_interhemispherical_asymmetry(eFC, method='absolute', num_regions=116):
    left_hemisphere_indices = np.arange(0, num_regions, 2)
    right_hemisphere_indices = np.arange(1, num_regions, 2)
    left_to_right = eFC[np.ix_(left_hemisphere_indices, right_hemisphere_indices)]
    right_to_left = eFC[np.ix_(right_hemisphere_indices, left_hemisphere_indices)]
    if method == 'index':
        asymmetry = (left_to_right - right_to_left) / (left_to_right + right_to_left)
    else:
        asymmetry = np.abs(left_to_right - right_to_left)
    return asymmetry


# Perform statistical analysis
def perform_statistical_analysis(asymmetry_patients, asymmetry_controls, threshold=0.01):
    # Flatten asymmetry matrices for comparison
    asymmetry_patients_flat = asymmetry_patients.reshape(asymmetry_patients.shape[0], -1)
    asymmetry_controls_flat = asymmetry_controls.reshape(asymmetry_controls.shape[0], -1)

    # Perform t-test
    t_stat, p_value = ttest_ind(asymmetry_patients_flat, asymmetry_controls_flat, axis=0)

    # Find significant differences
    significant_differences = p_value < threshold
    return t_stat, p_value, significant_differences

def suppress_interhemispheric(matrix):
    num_regions = matrix.shape[0] // 2
    for i in range(num_regions):
        left_region_index = 2 * i
        right_region_index = 2 * i + 1
        for j in range(num_regions):
            if j != i:
                left_region_other = 2 * j
                right_region_other = 2 * j + 1
                matrix[left_region_index, right_region_other] = 0
                matrix[right_region_index, left_region_other] = 0
    return matrix

def isHomotopic(i, j):
    return i//2 == j//2


###################### AAL116 MATH