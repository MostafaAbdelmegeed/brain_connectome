import numpy as np


def yeo_7_network():
    visual = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    somatomotor = [0, 1, 16, 17, 18, 19, 56, 57, 68, 69]
    dorsal_attention = [58, 59, 60, 61, 6, 7]
    ventral_attention = [28, 29, 62, 63]
    limbic = [36, 37, 38, 39, 40, 41, 24, 25, 20, 21]
    frontoparietal = [2, 3, 6, 7, 10, 11, 12, 13]
    default_mode = [22, 23, 32, 33, 34, 35, 64, 65, 66, 67, 84, 85]
    functional_groups = {
        'visual': visual,
        'somatomotor': somatomotor,
        'dorsal_attention': dorsal_attention,
        'ventral_attention': ventral_attention,
        'limbic': limbic,
        'frontoparietal': frontoparietal,
        'default_mode': default_mode,
        'other': []
    }
    all_assigned_nodes = (visual + somatomotor + dorsal_attention +
                          ventral_attention + limbic + frontoparietal + default_mode)
    for i in range(116):
        if i not in all_assigned_nodes:
            functional_groups['other'].append(i)
    return functional_groups



def compute_aal_to_yeo17_mapping(aal_img, yeo_img, aal_labels):
    # Get data arrays
    aal_data = aal_img.get_fdata()
    yeo_data = yeo_img.get_fdata()
    
    # Initialize the mapping dictionary
    aal_to_yeo17 = {}
    
    # Loop over each AAL ROI
    for i, label in enumerate(aal_labels):
        # AAL regions are labeled from 1 to 116
        aal_region_mask = (aal_data == (i + 1))
        
        # Skip if the region is empty
        if not aal_region_mask.any():
            continue
        
        # Extract overlapping Yeo network labels
        overlapping_yeo_labels = yeo_data[aal_region_mask]
        
        # Count occurrences of each Yeo label in the AAL region
        unique, counts = np.unique(overlapping_yeo_labels, return_counts=True)
        
        # Remove background label (0)
        yeo_labels = unique[unique != 0]
        yeo_counts = counts[unique != 0]
        
        if yeo_labels.size == 0:
            # No overlap with any Yeo network
            aal_to_yeo17[i] = 0  # Assign to 'other' or background
        else:
            # Assign to the Yeo network with the largest overlap
            max_index = np.argmax(yeo_counts)
            aal_to_yeo17[i] = int(yeo_labels[max_index])
    
    return aal_to_yeo17


def create_functional_groups(aal_to_yeo17_mapping):
    # Yeo networks are labeled from 1 to 17
    yeo_networks_17 = [
        'Visual1', 'Visual2', 'Visual3',
        'Somatomotor1', 'Somatomotor2',
        'DorsalAttention1', 'DorsalAttention2',
        'Salience1', 'Salience2',
        'Limbic_TempPole', 'Limbic_OFC',
        'Control1', 'Control2', 'Control3',
        'Default1', 'Default2', 'Default3'
    ]
    
    # Initialize the functional groups dictionary
    functional_groups = {name: [] for name in yeo_networks_17}
    functional_groups['other'] = []
    
    # Map AAL ROIs to Yeo network names
    for aal_index, yeo_label in aal_to_yeo17_mapping.items():
        if yeo_label == 0:
            functional_groups['other'].append(aal_index)
        else:
            network_name = yeo_networks_17[yeo_label - 1]
            functional_groups[network_name].append(aal_index)
    
    return functional_groups



def yeo_17_network():
    import numpy as np
    from nilearn import datasets, image
    from nilearn.image import resample_to_img

    # Load AAL atlas
    aal_atlas = datasets.fetch_atlas_aal()
    aal_labels = aal_atlas['labels']
    aal_img = image.load_img(aal_atlas['maps'])

    # Load Yeo 17 network atlas
    yeo_atlas = datasets.fetch_atlas_yeo_2011()
    yeo_17_img = yeo_atlas['thick_17']

    # Resample Yeo atlas to the AAL atlas space
    yeo_17_resampled = resample_to_img(yeo_17_img, aal_img, interpolation='nearest')

    # Compute the mapping
    aal_to_yeo17_mapping = compute_aal_to_yeo17_mapping(aal_img, yeo_17_resampled, aal_labels)

    # Create functional groups
    functional_groups = create_functional_groups(aal_to_yeo17_mapping)

    return functional_groups

