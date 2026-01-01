"""DINOv2 feature extraction module for GFGW-FM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class DINOv2FeatureExtractor(nn.Module):
    """
    DINOv2 feature extractor for semantic and texture features.

    This module extracts features from DINOv2 ViT models for computing
    the Fused Gromov-Wasserstein distance in GFGW-FM.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        feature_dim: int = 384,
        layer_index: int = -1,
        normalize_features: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize DINOv2 feature extractor.

        Args:
            model_name: DINOv2 model variant (dinov2_vits14, dinov2_vitb14, dinov2_vitl14)
            feature_dim: Output feature dimension
            layer_index: Which transformer layer to extract features from (-1 for last)
            normalize_features: Whether to L2-normalize output features
            device: Device to run the model on
        """
        super().__init__()

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.layer_index = layer_index
        self.normalize_features = normalize_features
        self.device = device

        # Load pretrained DINOv2 model
        self.model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True
        ).to(device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Image preprocessing
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        )

        # Patch size for DINOv2
        self.patch_size = 14

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess images for DINOv2.

        Args:
            images: Input images in range [-1, 1] with shape (B, C, H, W)

        Returns:
            Preprocessed images
        """
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Resize to multiple of patch size if needed
        h, w = images.shape[2:]
        target_h = (h // self.patch_size) * self.patch_size
        target_w = (w // self.patch_size) * self.patch_size

        if h != target_h or w != target_w:
            # Resize to at least 224x224 for better features
            target_size = max(224, target_h, target_w)
            target_size = (target_size // self.patch_size) * self.patch_size
            images = F.interpolate(
                images,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )

        # Normalize with ImageNet stats
        images = (images - self.mean) / self.std
        return images

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        return_patch_features: bool = False
    ) -> torch.Tensor:
        """
        Extract DINOv2 features from images.

        Args:
            images: Input images in range [-1, 1] with shape (B, C, H, W)
            return_patch_features: If True, return all patch tokens; else return CLS token

        Returns:
            Feature tensor of shape (B, feature_dim) or (B, num_patches, feature_dim)
        """
        images = self.preprocess(images)

        if return_patch_features:
            # Get all patch tokens (excluding CLS)
            features = self.model.forward_features(images)
            patch_tokens = features["x_norm_patchtokens"]
            if self.normalize_features:
                patch_tokens = F.normalize(patch_tokens, dim=-1)
            return patch_tokens
        else:
            # Get CLS token
            features = self.model(images)
            if self.normalize_features:
                features = F.normalize(features, dim=-1)
            return features

    @torch.no_grad()
    def extract_multi_scale_features(
        self,
        images: torch.Tensor,
        scales: List[int] = [1, 2, 4]
    ) -> torch.Tensor:
        """
        Extract multi-scale features for better texture representation.

        Args:
            images: Input images in range [-1, 1]
            scales: List of downscaling factors

        Returns:
            Concatenated multi-scale features
        """
        all_features = []

        for scale in scales:
            if scale > 1:
                scaled_images = F.avg_pool2d(images, kernel_size=scale)
            else:
                scaled_images = images

            features = self.forward(scaled_images)
            all_features.append(features)

        # Concatenate and project
        combined = torch.cat(all_features, dim=-1)
        return combined


class GlobalFeatureMemoryBank:
    """
    Global memory bank for storing dataset features.

    This class maintains a memory bank of DINOv2 features for the entire
    training dataset, enabling global OT computation instead of minibatch OT.
    """

    def __init__(
        self,
        feature_dim: int,
        max_size: int,
        device: torch.device = torch.device("cuda"),
        update_momentum: float = 0.0,  # 0 = no momentum (instant update)
    ):
        """
        Initialize the global feature memory bank.

        Args:
            feature_dim: Dimension of features to store
            max_size: Maximum number of features to store
            device: Device to store features on
            update_momentum: Momentum for feature updates (0 = replace, 1 = keep)
        """
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.device = device
        self.update_momentum = update_momentum

        # Initialize empty memory bank
        self.features = torch.zeros(max_size, feature_dim, device=device)
        self.valid_mask = torch.zeros(max_size, dtype=torch.bool, device=device)
        self.indices = torch.zeros(max_size, dtype=torch.long, device=device)
        self.num_valid = 0

        # Precomputed distance matrix for structure term (updated periodically)
        self._distance_matrix = None
        self._distance_matrix_valid = False

    @torch.no_grad()
    def initialize_from_dataset(
        self,
        feature_extractor: DINOv2FeatureExtractor,
        dataloader: torch.utils.data.DataLoader,
        max_samples: Optional[int] = None
    ):
        """
        Initialize memory bank with features from entire dataset.

        Args:
            feature_extractor: DINOv2 feature extractor
            dataloader: DataLoader for the dataset
            max_samples: Maximum number of samples to process (None = all)
        """
        all_features = []
        all_indices = []
        count = 0

        for batch_idx, batch_data in enumerate(dataloader):
            # Handle both (images, labels) and (images, labels, indices) formats
            images = batch_data[0]
            if max_samples is not None and count >= max_samples:
                break

            images = images.to(self.device).float()
            if images.max() > 1:
                images = images / 127.5 - 1  # Convert from [0, 255] to [-1, 1]

            features = feature_extractor(images)

            batch_size = features.shape[0]
            indices = torch.arange(
                count,
                count + batch_size,
                device=self.device
            )

            all_features.append(features)
            all_indices.append(indices)
            count += batch_size

            if count >= self.max_size:
                break

        # Concatenate and store
        all_features = torch.cat(all_features, dim=0)[:self.max_size]
        all_indices = torch.cat(all_indices, dim=0)[:self.max_size]

        self.num_valid = min(count, self.max_size)
        self.features[:self.num_valid] = all_features[:self.num_valid]
        self.indices[:self.num_valid] = all_indices[:self.num_valid]
        self.valid_mask[:self.num_valid] = True

        # Invalidate cached distance matrix
        self._distance_matrix_valid = False

        print(f"Initialized memory bank with {self.num_valid} features")

    @torch.no_grad()
    def update_features(
        self,
        new_features: torch.Tensor,
        indices: torch.Tensor
    ):
        """
        Update features in the memory bank.

        Args:
            new_features: New feature vectors (B, feature_dim)
            indices: Indices in the dataset for these features
        """
        for i, idx in enumerate(indices):
            if idx < self.max_size:
                if self.update_momentum > 0 and self.valid_mask[idx]:
                    self.features[idx] = (
                        self.update_momentum * self.features[idx] +
                        (1 - self.update_momentum) * new_features[i]
                    )
                else:
                    self.features[idx] = new_features[i]
                    self.valid_mask[idx] = True
                    self.indices[idx] = idx

        self.num_valid = self.valid_mask.sum().item()
        self._distance_matrix_valid = False

    def get_valid_features(self) -> torch.Tensor:
        """Get all valid features from the memory bank."""
        return self.features[:self.num_valid]

    @torch.no_grad()
    def compute_distance_matrix(
        self,
        use_cosine: bool = False,
        batch_size: int = 1024
    ) -> torch.Tensor:
        """
        Compute pairwise distance matrix for structure term in FGW.

        Args:
            use_cosine: Use cosine distance instead of Euclidean
            batch_size: Batch size for computation to manage memory

        Returns:
            Distance matrix of shape (num_valid, num_valid)
        """
        if self._distance_matrix_valid and self._distance_matrix is not None:
            return self._distance_matrix

        n = self.num_valid
        features = self.features[:n]

        # Compute in batches to manage memory
        distance_matrix = torch.zeros(n, n, device=self.device)

        for i in range(0, n, batch_size):
            end_i = min(i + batch_size, n)
            for j in range(0, n, batch_size):
                end_j = min(j + batch_size, n)

                if use_cosine:
                    # Cosine distance
                    sim = torch.mm(features[i:end_i], features[j:end_j].T)
                    distance_matrix[i:end_i, j:end_j] = 1 - sim
                else:
                    # Euclidean distance
                    diff = features[i:end_i].unsqueeze(1) - features[j:end_j].unsqueeze(0)
                    distance_matrix[i:end_i, j:end_j] = (diff ** 2).sum(dim=-1).sqrt()

        self._distance_matrix = distance_matrix
        self._distance_matrix_valid = True

        return distance_matrix

    def sample_subset(
        self,
        size: int,
        return_distances: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a subset of features for batch OT computation.

        Args:
            size: Number of samples to draw
            return_distances: Whether to return pairwise distances

        Returns:
            Tuple of (sampled_features, sampled_indices) or
            (sampled_features, sampled_indices, sampled_distances)
        """
        if size >= self.num_valid:
            indices = torch.arange(self.num_valid, device=self.device)
        else:
            indices = torch.randperm(self.num_valid, device=self.device)[:size]

        features = self.features[indices]

        if return_distances:
            # Compute pairwise distances for the subset
            if self._distance_matrix_valid:
                distances = self._distance_matrix[indices][:, indices]
            else:
                distances = torch.cdist(features, features)
            return features, indices, distances

        return features, indices
