import graphIO
import sys
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python generate_attributes_from_adj_matrix.py <matrix_filepath> <mappings_filepath> <asymmetry_method>')
        sys.exit(1)

    file_path = sys.argv[1]
    mappings_file_path = sys.argv[2]
    ASYMMETRY_METHOD = sys.argv[3]

    filename = os.path.basename(file_path)
    # Read adjacency matrices from the directory
    adj_matrix = graphIO.read_adj_matrix_from_file(file_path)
    region_name_mappings = graphIO.read_mappings_from_json(mappings_file_path)
    reverse_region_name_mappings = {
        v: k for k, v in region_name_mappings.items()}
    shape = (len(region_name_mappings.keys()),
             len(region_name_mappings.keys()))
    if adj_matrix.shape != shape:
        raise ValueError(f'AD matrix shape: {adj_matrix.shape}')
    lh_regions = []
    rh_regions = []
    for value in region_name_mappings.values():
        if 'lh' in value:
            lh_regions.append(value)
        else:
            rh_regions.append(value)
    print(f'lh regions: {len(lh_regions)}, rh regions: {len(rh_regions)}')
    for lh_region in lh_regions:
        lh_region_test = lh_region.replace('lh', 'rh')
        if lh_region_test not in rh_regions:
            raise ValueError(f"No corresponding region found for {lh_region}")
    print('All Left Hemisphere regions have corresponding Right Hemisphere regions')
    lh_curves = []
    rh_curves = []
    abs_diffs = []

    for lh_region, rh_region in zip(lh_regions, rh_regions):
        if ASYMMETRY_METHOD == "abs_diff":
            lh_curves.append(
                adj_matrix[reverse_region_name_mappings[lh_region]-1].sum())
            rh_curves.append(
                adj_matrix[reverse_region_name_mappings[rh_region]-1].sum())
            abs_diffs.append(
                abs(lh_curves[-1] - rh_curves[-1]))

    abs_diffs = np.array(abs_diffs)
    resized_diffs = np.zeros((shape[0], 1))
    for i in range(len(abs_diffs)):
        resized_diffs[reverse_region_name_mappings[lh_regions[i]]-1] = abs_diffs[i]
        resized_diffs[reverse_region_name_mappings[rh_regions[i]]-1] = abs_diffs[i]
    region_labels = [region.replace('_lh_', '') for region in lh_regions]
    regions_indices = [(reverse_region_name_mappings[lh_regions[i]]-1, reverse_region_name_mappings[rh_regions[i]]-1)
                       for i in range(len(lh_regions))]
    attributes = np.zeros((shape[0], 2))
    print(shape)
    print(attributes.shape)
    print(len(region_labels))
    attributes[:, 0] = resized_diffs.reshape(-1)
    for tuple in regions_indices:
        attributes[tuple[0], 1] = tuple[0]
        attributes[tuple[1], 1] = tuple[0]
    df = pd.DataFrame(attributes, columns=['abs_asymmetry', 'region'])
    df.to_csv(f'{filename}.csv', index=False, mode='w')
