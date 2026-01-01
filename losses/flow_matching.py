"""Loss functions for GFGW-FM training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class FlowMatchingLoss(nn.Module):
    """
    Flow matching loss for training the one-step generator.

    This loss trains the generator to match the OT-optimal transport map
    by regressing the generated samples to their matched targets.
    """

    def __init__(
        self,
        loss_type: str = 'l2',
        sigma_data: float = 0.5,
        use_huber: bool = False,
        huber_c: float = 0.0,
    ):
        """
        Initialize flow matching loss.

        Args:
            loss_type: Type of loss ('l2', 'l1', 'lpips')
            sigma_data: Data standard deviation for weighting
            use_huber: Use Huber loss variant
            huber_c: Huber loss parameter
        """
        super().__init__()
        self.loss_type = loss_type
        self.sigma_data = sigma_data
        self.use_huber = use_huber
        self.huber_c = huber_c

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.

        Args:
            generated: Generated images (B, C, H, W)
            target: Target images (B, C, H, W) matched via OT
            weights: Optional per-sample weights (B,)

        Returns:
            Scalar loss value
        """
        if self.loss_type == 'l2':
            # L2 loss (MSE)
            diff = generated - target
            per_sample_loss = (diff ** 2).view(diff.shape[0], -1).sum(dim=-1)
        elif self.loss_type == 'l1':
            # L1 loss
            diff = generated - target
            per_sample_loss = diff.abs().view(diff.shape[0], -1).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Apply Huber-like transformation if enabled
        if self.use_huber and self.huber_c > 0:
            per_sample_loss = torch.sqrt(per_sample_loss + self.huber_c ** 2) - self.huber_c
        else:
            per_sample_loss = torch.sqrt(per_sample_loss + 1e-8)

        # Apply weights
        if weights is not None:
            per_sample_loss = per_sample_loss * weights

        return per_sample_loss.mean()


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss to enforce semantic consistency.

    This loss encourages the generated images to have similar
    DINOv2 features to their matched targets.
    """

    def __init__(
        self,
        feature_dim: int = 384,
        use_cosine: bool = True,
    ):
        """
        Initialize feature matching loss.

        Args:
            feature_dim: Dimension of features
            use_cosine: Use cosine similarity instead of L2
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.use_cosine = use_cosine

    def forward(
        self,
        features_gen: torch.Tensor,
        features_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            features_gen: Generated sample features (B, d)
            features_target: Target sample features (B, d)

        Returns:
            Scalar loss value
        """
        if self.use_cosine:
            # Cosine distance
            features_gen = F.normalize(features_gen, dim=-1)
            features_target = F.normalize(features_target, dim=-1)
            similarity = (features_gen * features_target).sum(dim=-1)
            loss = (1 - similarity).mean()
        else:
            # L2 distance
            diff = features_gen - features_target
            loss = (diff ** 2).sum(dim=-1).mean()

        return loss


class GFGWFlowMatchingLoss(nn.Module):
    """
    Complete GFGW-FM loss combining flow matching with FGW-based OT.

    This is the main loss function for training GFGW-FM, implementing:
    1. OT-based sample matching using Fused Gromov-Wasserstein
    2. Flow matching regression loss to matched targets
    3. Optional feature consistency loss
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        flow_loss_weight: float = 1.0,
        feature_loss_weight: float = 0.1,
        use_huber: bool = False,
        huber_c: float = 0.0,
    ):
        """
        Initialize GFGW-FM loss.

        Args:
            sigma_data: Data standard deviation
            flow_loss_weight: Weight for flow matching loss
            feature_loss_weight: Weight for feature matching loss
            use_huber: Use Huber loss variant
            huber_c: Huber loss parameter
        """
        super().__init__()
        self.sigma_data = sigma_data
        self.flow_loss_weight = flow_loss_weight
        self.feature_loss_weight = feature_loss_weight

        self.flow_loss = FlowMatchingLoss(
            loss_type='l2',
            sigma_data=sigma_data,
            use_huber=use_huber,
            huber_c=huber_c,
        )
        self.feature_loss = FeatureMatchingLoss(use_cosine=True)

    def forward(
        self,
        generated: torch.Tensor,
        matched_targets: torch.Tensor,
        features_gen: Optional[torch.Tensor] = None,
        features_target: Optional[torch.Tensor] = None,
        coupling: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GFGW-FM loss.

        Args:
            generated: Generated images (B, C, H, W)
            matched_targets: OT-matched target images (B, C, H, W)
            features_gen: Generated sample features (B, d)
            features_target: Target sample features (B, d)
            coupling: OT coupling matrix (B, M) for soft matching

        Returns:
            Dictionary containing individual loss terms and total loss
        """
        losses = {}

        # Flow matching loss (main term)
        flow_loss = self.flow_loss(generated, matched_targets)
        losses['flow_loss'] = flow_loss

        # Feature matching loss
        if features_gen is not None and features_target is not None:
            feat_loss = self.feature_loss(features_gen, features_target)
            losses['feature_loss'] = feat_loss
        else:
            feat_loss = torch.tensor(0.0, device=generated.device)
            losses['feature_loss'] = feat_loss

        # Total loss
        total_loss = (
            self.flow_loss_weight * flow_loss +
            self.feature_loss_weight * feat_loss
        )
        losses['total_loss'] = total_loss

        return losses


class TextureConsistencyLoss(nn.Module):
    """
    Texture Consistency Loss for encouraging texture preservation.

    This loss compares local patch statistics between generated and
    target images to encourage texture fidelity.
    """

    def __init__(
        self,
        patch_size: int = 8,
        num_patches: int = 64,
    ):
        """
        Initialize texture consistency loss.

        Args:
            patch_size: Size of patches to extract
            num_patches: Number of random patches per image
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def extract_random_patches(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract random patches from images.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Patches (B, num_patches, C, patch_size, patch_size)
        """
        B, C, H, W = images.shape
        patches = []

        for _ in range(self.num_patches):
            y = torch.randint(0, H - self.patch_size + 1, (B,), device=images.device)
            x = torch.randint(0, W - self.patch_size + 1, (B,), device=images.device)

            # Extract patches
            batch_patches = []
            for b in range(B):
                patch = images[b, :, y[b]:y[b]+self.patch_size, x[b]:x[b]+self.patch_size]
                batch_patches.append(patch)
            patches.append(torch.stack(batch_patches, dim=0))

        patches = torch.stack(patches, dim=1)  # (B, num_patches, C, ps, ps)
        return patches

    def compute_patch_statistics(
        self,
        patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance statistics for patches.

        Args:
            patches: Patches (B, num_patches, C, ps, ps)

        Returns:
            Tuple of (mean, variance) statistics
        """
        # Flatten patches
        B, N, C, H, W = patches.shape
        patches_flat = patches.view(B, N, -1)

        mean = patches_flat.mean(dim=-1)  # (B, N)
        var = patches_flat.var(dim=-1)  # (B, N)

        return mean, var

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute texture consistency loss.

        Args:
            generated: Generated images (B, C, H, W)
            target: Target images (B, C, H, W)

        Returns:
            Scalar loss value
        """
        # Extract patches
        gen_patches = self.extract_random_patches(generated)
        tgt_patches = self.extract_random_patches(target)

        # Compute statistics
        gen_mean, gen_var = self.compute_patch_statistics(gen_patches)
        tgt_mean, tgt_var = self.compute_patch_statistics(tgt_patches)

        # Compute loss (difference in statistics)
        mean_loss = ((gen_mean - tgt_mean) ** 2).mean()
        var_loss = ((gen_var - tgt_var) ** 2).mean()

        return mean_loss + var_loss
