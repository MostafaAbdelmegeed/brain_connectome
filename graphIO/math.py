import numpy as np

###################### AAL116 MATH

def calculate_interhemispheric(matrix, method='abs_diff', epsilon=1e-10):
    means = np.mean(matrix, axis=1)
    ai = []
    for i in range(0, len(means)-1, 2):
        if method == 'abs_diff':
            ai.append(np.abs(means[i] - means[i+1]))
        elif method == 'ai':
            ai.append((means[i] - means[i+1]) / (means[i] + means[i+1] + epsilon))
        elif method == 'sq_diff':
            ai.append((means[i] - means[i+1])**2)
        elif method == 'log_ratio':
            ai.append(np.log((means[i] + epsilon) / (means[i+1] + epsilon)))
        elif method == 'perc_diff':
            ai.append((means[i] - means[i+1]) / ((means[i] + means[i+1] + epsilon) / 2) * 100)
        elif method == 'ratio':
            ai.append((means[i] + epsilon) / (means[i+1] + epsilon))
        elif method == 'smape':
            ai.append(100 * (np.abs(means[i] - means[i+1]) / ((np.abs(means[i]) + np.abs(means[i+1]) + epsilon) / 2)))
    return ai

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