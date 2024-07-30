import torch

def yeo_network(as_tensor=False):
    # Define the refined functional classes based on Yeo's 7 Network Parcellations
    visual = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    somatomotor = [0, 1, 57, 58, 69, 70, 16, 17, 18, 19]
    dorsal_attention = [59, 60, 61, 62, 67, 68, 6, 7]
    ventral_attention = [12, 13, 63, 64, 29, 30]
    limbic = [37, 38, 39, 40, 41, 42, 31, 32, 33, 34, 35, 36]
    frontoparietal = [2, 3, 4, 5, 6, 7, 10, 11]
    default_mode = [23, 24, 67, 68, 65, 66, 35, 36]
    # Initialize the one-hot encoding list
    one_hot_encodings = []
    # Create one-hot encodings for each region
    for i in range(116):
        encoding = [
            1 if i in visual else 0,
            1 if i in somatomotor else 0,
            1 if i in dorsal_attention else 0,
            1 if i in ventral_attention else 0,
            1 if i in limbic else 0,
            1 if i in frontoparietal else 0,
            1 if i in default_mode else 0
        ]
        one_hot_encodings.append(encoding)
    if as_tensor:
        return torch.tensor(one_hot_encodings, dtype=torch.float)
    else:
        return one_hot_encodings