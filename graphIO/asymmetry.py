import numpy as np

def calculate_inter_hemispheric_asymmetry_vector(matrix, method='abs_diff', atlas='aal116'):
    """
    Calculates the asymmetry vector based on the given matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix for calculating the asymmetry vector.
    - method (str): The method used for calculating the asymmetry vector. Default is 'abs_diff'.
    - atlas (str): The atlas used for calculating the asymmetry vector. Default is 'aal116'.

    Returns:
    - numpy.ndarray: The calculated asymmetry vector.

    Raises:
    - ValueError: If the provided atlas is invalid.
    """
    if atlas=='aal116':
        return calculate_inter_hemispheric_asymmetry_vector_aal116(matrix, method)
    else:
        raise ValueError('Invalid atlas')


def calculate_inter_hemispheric_asymmetry_vector_aal116(matrix, method='abs_diff'):
    """
    Calculates the inter-hemispheric asymmetry vector for a given matrix.

    Parameters:
    - matrix (numpy.ndarray): The input matrix for which the asymmetry vector needs to be calculated.
    - method (str): The method used to calculate the asymmetry vector. Default is 'abs_diff'.

    Returns:
    - ai (list): The calculated inter-hemispheric asymmetry vector.

    Available methods:
    - 'abs_diff': Calculates the absolute difference between the means of each pair of hemispheres.
    - 'ai': Calculates the asymmetry index as the difference between the means divided by the sum of the means.
    - 'sq_diff': Calculates the squared difference between the means of each pair of hemispheres.
    - 'log_ratio': Calculates the logarithm of the ratio between the means of each pair of hemispheres.
    - 'perc_diff': Calculates the percentage difference between the means of each pair of hemispheres.
    - 'ratio': Calculates the ratio between the means of each pair of hemispheres.
    - 'smape': Calculates the symmetric mean absolute percentage error between the means of each pair of hemispheres.
    """
    means = np.mean(matrix, axis=1)
    ai = []
    for i in range(0, len(means)-1, 2):
        if method == 'abs_diff':
            ai.append(np.abs(means[i] - means[i+1]))
        elif method == 'ai':
            ai.append((means[i] - means[i+1]) / (means[i] + means[i+1]))
        elif method == 'sq_diff':
            ai.append((means[i] - means[i+1])**2)
        elif method == 'log_ratio':
            ai.append(np.log(means[i] / means[i+1]))
        elif method == 'perc_diff':
            ai.append((means[i] - means[i+1]) / ((means[i] + means[i+1]) / 2) * 100)
        elif method == 'ratio':
            ai.append(means[i] / means[i+1])
        elif method == 'smape':
            ai.append(100 * (np.abs(means[i] - means[i+1]) / ((np.abs(means[i]) + np.abs(means[i+1])) / 2)))
    return ai