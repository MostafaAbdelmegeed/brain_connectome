import torch
from .preprocess import center_matrices

def load_data(dataset_path):
    ppmi_dataset = torch.load(dataset_path)
    connectivity_matrices = center_matrices(ppmi_dataset['data'].numpy())
    connectivity_labels = ppmi_dataset['class_label'].numpy().reshape(-1, 1)
    return connectivity_matrices, connectivity_labels