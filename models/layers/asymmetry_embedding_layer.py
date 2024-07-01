import torch
import torch.nn as nn
import numpy as np

class HemisphericalAsymmetryEmbeddingLayer(nn.Module):
    def __init__(self, num_regions, threshold, mask_type='both'):
        super(HemisphericalAsymmetryEmbeddingLayer, self).__init__()
        self.num_regions = num_regions
        self.threshold = threshold
        self.mask_type = mask_type
        self.left_indices = np.arange(0, num_regions, 2)
        self.right_indices = np.arange(1, num_regions, 2)
        self.create_masks()

    def create_masks(self):
        # Create intrahemispheric mask
        self.intra_mask = torch.zeros((self.num_regions, self.num_regions), dtype=torch.float32)
        self.intra_mask[np.ix_(self.left_indices, self.left_indices)] = 1
        self.intra_mask[np.ix_(self.right_indices, self.right_indices)] = 1

        # Create interhemispheric mask
        self.inter_mask = torch.zeros((self.num_regions, self.num_regions), dtype=torch.float32)
        self.inter_mask[np.ix_(self.left_indices, self.right_indices)] = 1
        self.inter_mask[np.ix_(self.right_indices, self.left_indices)] = 1

    def forward(self, eFC):
        # Suppress noisy connections
        eFC = torch.where(eFC >= self.threshold, eFC, torch.tensor(0.0, device=eFC.device))

        # Mark connections based on mask type
        if self.mask_type == 'intra':
            # Only keep intrahemispheric connections
            masked_connections = eFC * self.intra_mask.to(eFC.device)
        elif self.mask_type == 'inter':
            # Only keep interhemispheric connections
            masked_connections = eFC * self.inter_mask.to(eFC.device)
        elif self.mask_type == 'both':
            # Keep both intrahemispheric and interhemispheric connections above the threshold
            intra_connections = eFC * self.intra_mask.to(eFC.device)
            inter_connections = eFC * self.inter_mask.to(eFC.device)
            masked_connections = intra_connections + inter_connections

        return masked_connections

# Example usage
num_regions = 116
threshold = 0.7
eFC = torch.rand((num_regions, num_regions))

# Create embedding layer with different mask types
embedding_layer_intra = HemisphericalAsymmetryEmbeddingLayer(num_regions, threshold, mask_type='intra')
embedding_layer_inter = HemisphericalAsymmetryEmbeddingLayer(num_regions, threshold, mask_type='inter')
embedding_layer_both = HemisphericalAsymmetryEmbeddingLayer(num_regions, threshold, mask_type='both')

masked_connections_intra = embedding_layer_intra(eFC)
masked_connections_inter = embedding_layer_inter(eFC)
masked_connections_both = embedding_layer_both(eFC)

print("Masked Connections (Intra) Shape:", masked_connections_intra.shape)
print("Masked Connections (Inter) Shape:", masked_connections_inter.shape)
print("Masked Connections (Both) Shape:", masked_connections_both.shape)

print("Non zero elements (Intra):", torch.nonzero(masked_connections_intra).shape[0]/masked_connections_intra.shape[0])
print("Non zero elements (Inter):", torch.nonzero(masked_connections_inter).shape[0]/masked_connections_intra.shape[0])
print("Non zero elements (Both):", torch.nonzero(masked_connections_both).shape[0]/masked_connections_intra.shape[0])

