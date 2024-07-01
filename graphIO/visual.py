import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_asymmetry_comparison(asymmetry_patients, asymmetry_controls, significant_differences, title_suffix, left_indices, right_indices, threshold=0.01, dataset=''):
    # Average asymmetry for each group
    mean_asymmetry_patients = np.mean(asymmetry_patients, axis=0)
    mean_asymmetry_controls = np.mean(asymmetry_controls, axis=0)
    
    # Plot average asymmetry and significant differences
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f'{dataset} eFC Asymmetry - {title_suffix}', fontsize=16)

    # Plot average asymmetry for patients
    sns.heatmap(mean_asymmetry_patients, ax=axes[0], cmap='coolwarm', cbar=True)
    axes[0].set_title('Average Asymmetry - Patients')

    # Plot average asymmetry for controls
    sns.heatmap(mean_asymmetry_controls, ax=axes[1], cmap='coolwarm', cbar=True)
    axes[1].set_title('Average Asymmetry - Controls')

    # Plot significant differences
    significant_diff_matrix = significant_differences.reshape(len(left_indices), len(right_indices))
    sns.heatmap(significant_diff_matrix, ax=axes[2], cmap='coolwarm', cbar=True)
    axes[2].set_title(f'Significant Differences (p < {threshold})')

    # Adjust layout
    plt.tight_layout()
    plt.show()