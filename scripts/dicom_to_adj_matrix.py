import os
from nilearn import datasets, input_data, image
from nilearn.connectome import ConnectivityMeasure
import numpy as np

def convert_dicom_to_nifti(dicom_dir, nifti_output_dir):
    # Use dcm2niix to convert DICOM to NIfTI
    os.system(f'dcm2niix -o {nifti_output_dir} {dicom_dir}')

# Convert your DICOM directory to NIfTI
dicom_dir = 'data/ppmi_dicom'
nifti_output_dir = 'data/ppmi_nifti'
convert_dicom_to_nifti(dicom_dir, nifti_output_dir)

# Load the converted NIfTI file
nifti_files = [f for f in os.listdir(nifti_output_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
nifti_file = os.path.join(nifti_output_dir, nifti_files[0])
print(f"Using NIfTI file: {nifti_file}")

# Load AAL atlas
aal_atlas = datasets.fetch_atlas_aal()

# Inspect the keys in the fetched atlas object
print(f"Available keys in aal_atlas: {aal_atlas.keys()}")

# Use the correct key for the atlas filename
atlas_filename = aal_atlas.maps  # This is usually the key for the atlas image file
print(f"Using AAL atlas file: {atlas_filename}")

# Extract time series for each ROI in the AAL atlas
masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
time_series = masker.fit_transform(nifti_file)

print(f"Extracted time series shape: {time_series.shape}")
# Shape should be (n_timepoints, n_regions) where n_regions should be 116 for AAL atlas

# Compute correlation matrix
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

print(f"Correlation matrix shape: {correlation_matrix.shape}")
# Shape should be (n_regions, n_regions), which should be (116, 116) for AAL atlas

# (Optional) Preprocess the connectivity matrix
def preprocess_adjacency_matrix(adjacency_matrix, percent):
    top_percent = np.percentile(adjacency_matrix.flatten(), 100 - percent)
    adjacency_matrix[adjacency_matrix < top_percent] = 0
    return adjacency_matrix

# Example: keep top 10% values
adjacency_matrix = preprocess_adjacency_matrix(correlation_matrix, 10)
print(f"Processed adjacency matrix shape: {adjacency_matrix.shape}")

# Save the processed adjacency matrix if needed
np.save('data/ppmi_nifti/adjacency_matrix.npy', adjacency_matrix)
