"""Enhanced Flow Matching Loss for GFGW-FM.

Integrates all advanced loss components from state-of-the-art papers:
- ECM: Pseudo-Huber loss, adaptive weighting
- SlimFlow: LPIPS perceptual loss, multi-scale loss
- Boundary RF: Boundary condition loss
- TCM: Consistency loss, structure preservation

Original loss classes maintained for backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np

# Try to import advanced losses
try:
    from losses.advanced_losses import (
        PseudoHuberLoss,
        LPIPSLoss,
        BoundaryConditionLoss,
        ConsistencyLoss,
        AdaptiveWeighting,
        StructurePreservationLoss,
    )
    HAS_ADVANCED = True
except ImportError:
    HAS_ADVANCED = False


# ============================================================================
# Original Loss Classes (maintained for backward compatibility)
# ============================================================================

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
        if self.loss_type == 'l2':
            diff = generated - target
            per_sample_loss = (diff ** 2).view(diff.shape[0], -1).sum(dim=-1)
        elif self.loss_type == 'l1':
            diff = generated - target
            per_sample_loss = diff.abs().view(diff.shape[0], -1).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        if self.use_huber and self.huber_c > 0:
            per_sample_loss = torch.sqrt(per_sample_loss + self.huber_c ** 2) - self.huber_c
        else:
            per_sample_loss = torch.sqrt(per_sample_loss + 1e-8)

        if weights is not None:
            per_sample_loss = per_sample_loss * weights

        return per_sample_loss.mean()


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss to enforce semantic consistency."""

    def __init__(
        self,
        feature_dim: int = 384,
        use_cosine: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_cosine = use_cosine

    def forward(
        self,
        features_gen: torch.Tensor,
        features_target: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_cosine:
            features_gen = F.normalize(features_gen, dim=-1)
            features_target = F.normalize(features_target, dim=-1)
            similarity = (features_gen * features_target).sum(dim=-1)
            loss = (1 - similarity).mean()
        else:
            diff = features_gen - features_target
            loss = (diff ** 2).sum(dim=-1).mean()

        return loss


class TextureConsistencyLoss(nn.Module):
    """Texture Consistency Loss for encouraging texture preservation."""

    def __init__(
        self,
        patch_size: int = 8,
        num_patches: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches

    def extract_random_patches(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        patches = []

        for _ in range(self.num_patches):
            y = torch.randint(0, H - self.patch_size + 1, (B,), device=images.device)
            x = torch.randint(0, W - self.patch_size + 1, (B,), device=images.device)

            batch_patches = []
            for b in range(B):
                patch = images[b, :, y[b]:y[b]+self.patch_size, x[b]:x[b]+self.patch_size]
                batch_patches.append(patch)
            patches.append(torch.stack(batch_patches, dim=0))

        patches = torch.stack(patches, dim=1)
        return patches

    def compute_patch_statistics(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = patches.shape
        patches_flat = patches.view(B, N, -1)

        mean = patches_flat.mean(dim=-1)
        var = patches_flat.var(dim=-1)

        return mean, var

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gen_patches = self.extract_random_patches(generated)
        tgt_patches = self.extract_random_patches(target)

        gen_mean, gen_var = self.compute_patch_statistics(gen_patches)
        tgt_mean, tgt_var = self.compute_patch_statistics(tgt_patches)

        mean_loss = ((gen_mean - tgt_mean) ** 2).mean()
        var_loss = ((gen_var - tgt_var) ** 2).mean()

        return mean_loss + var_loss


# ============================================================================
# Enhanced Loss Classes
# ============================================================================

class PseudoHuberLossSimple(nn.Module):
    """
    Pseudo-Huber loss - fallback implementation.

    Loss = sqrt(||x - y||^2 + c^2) - c
    """

    def __init__(self, c: float = 0.00054, reduction: str = "mean"):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = pred - target

        if diff.dim() > 2:
            l2_sq = (diff ** 2).flatten(1).sum(dim=1)
        else:
            l2_sq = (diff ** 2).sum(dim=1)

        # Scale c by dimension
        dim = diff[0].numel()
        c_scaled = self.c * np.sqrt(dim)

        if c_scaled > 0:
            loss = torch.sqrt(l2_sq + c_scaled ** 2) - c_scaled
        else:
            loss = torch.sqrt(l2_sq)

        if weight is not None:
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultiScaleFlowLoss(nn.Module):
    """
    Multi-scale flow matching loss for improved texture preservation.

    Computes loss at multiple resolutions.
    """

    def __init__(
        self,
        scales: Tuple[int, ...] = (1, 2, 4),
        scale_weights: Optional[Tuple[float, ...]] = None,
        use_pseudo_huber: bool = True,
        huber_c: float = 0.00054,
    ):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights or tuple([1.0 / len(scales)] * len(scales))

        if use_pseudo_huber:
            if HAS_ADVANCED:
                self.loss_fn = PseudoHuberLoss(c=huber_c)
            else:
                self.loss_fn = PseudoHuberLossSimple(c=huber_c)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1:
                gen_scaled = generated
                tgt_scaled = target
            else:
                gen_scaled = F.avg_pool2d(generated, kernel_size=scale)
                tgt_scaled = F.avg_pool2d(target, kernel_size=scale)

            if isinstance(self.loss_fn, (PseudoHuberLossSimple,)) or (HAS_ADVANCED and isinstance(self.loss_fn, PseudoHuberLoss)):
                loss = self.loss_fn(gen_scaled, tgt_scaled)
            else:
                loss = self.loss_fn(gen_scaled, tgt_scaled)
            total_loss = total_loss + weight * loss

        return total_loss


class AdaptiveWeightingSimple(nn.Module):
    """
    Adaptive loss weighting based on SNR - fallback implementation.
    """

    def __init__(
        self,
        weighting_type: str = "snr",
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.weighting_type = weighting_type
        self.sigma_data = sigma_data

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.weighting_type == "uniform":
            return torch.ones_like(t)

        sigma = t * self.sigma_data
        snr = (self.sigma_data ** 2) / (sigma ** 2 + 1e-8)

        if self.weighting_type == "snr":
            weight = snr / (snr + 1)
        elif self.weighting_type == "truncated_snr":
            weight = torch.clamp(snr / (snr + 1), min=0.1, max=10.0)
        else:
            weight = torch.ones_like(t)

        weight = weight / weight.mean()
        return weight


class BoundaryConditionLossSimple(nn.Module):
    """
    Boundary condition loss - fallback implementation.

    Enforces v(x, 1) = x (identity at t=1).
    """

    def __init__(self, huber_c: float = 0.00054):
        super().__init__()
        self.loss_fn = PseudoHuberLossSimple(c=huber_c)

    def forward(
        self,
        model_output: torch.Tensor,
        noise: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.0

        # End boundary: at t=1, output should equal noise
        end_mask = (t > 0.99).float()
        if end_mask.sum() > 0:
            end_loss = self.loss_fn(
                model_output * end_mask.view(-1, 1, 1, 1),
                noise * end_mask.view(-1, 1, 1, 1)
            )
            loss = loss + end_loss

        # Start boundary: at t=0, output should equal target
        start_mask = (t < 0.01).float()
        if start_mask.sum() > 0:
            start_loss = self.loss_fn(
                model_output * start_mask.view(-1, 1, 1, 1),
                target * start_mask.view(-1, 1, 1, 1)
            )
            loss = loss + start_loss

        return loss


class StructurePreservationLossSimple(nn.Module):
    """
    Structure preservation loss - fallback implementation.
    """

    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

    def forward(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        D_gen = torch.cdist(features_gen, features_gen, p=2)
        D_real = torch.cdist(features_real, features_real, p=2)

        if self.normalize:
            D_gen = D_gen / (D_gen.max() + 1e-8)
            D_real = D_real / (D_real.max() + 1e-8)

        D_real_transported = torch.mm(torch.mm(coupling, D_real), coupling.T)
        loss = F.mse_loss(D_gen, D_real_transported)

        return loss


# ============================================================================
# Main Enhanced Loss Class
# ============================================================================

class GFGWFlowMatchingLoss(nn.Module):
    """
    Complete GFGW-FM loss - enhanced version.

    Combines:
    1. Pseudo-Huber loss for robustness (from ECM/TCM)
    2. Multi-scale loss for texture preservation
    3. Feature matching loss for semantic consistency
    4. Optional: LPIPS, boundary loss, consistency loss, structure loss
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        flow_loss_weight: float = 1.0,
        feature_loss_weight: float = 0.1,
        use_huber: bool = True,
        huber_c: float = 0.00054,
        # New parameters
        use_lpips: bool = False,
        lpips_weight: float = 0.5,
        use_boundary: bool = False,
        boundary_weight: float = 0.1,
        use_structure: bool = False,
        structure_weight: float = 0.1,
        use_adaptive_weighting: bool = True,
        weighting_type: str = "snr",
        use_multiscale: bool = True,
    ):
        super().__init__()

        self.sigma_data = sigma_data
        self.flow_loss_weight = flow_loss_weight
        self.feature_loss_weight = feature_loss_weight

        # Primary loss functions
        if use_huber:
            if HAS_ADVANCED:
                self.recon_loss = PseudoHuberLoss(c=huber_c)
                self.feature_loss_fn = PseudoHuberLoss(c=huber_c * 0.1)
            else:
                self.recon_loss = PseudoHuberLossSimple(c=huber_c)
                self.feature_loss_fn = PseudoHuberLossSimple(c=huber_c * 0.1)
        else:
            self.recon_loss = nn.MSELoss()
            self.feature_loss_fn = nn.MSELoss()

        # Multi-scale loss
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale_loss = MultiScaleFlowLoss(
                scales=(1, 2, 4),
                use_pseudo_huber=use_huber,
                huber_c=huber_c,
            )

        # LPIPS perceptual loss
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        if use_lpips and HAS_ADVANCED:
            self.lpips_loss = LPIPSLoss(net="vgg")

        # Boundary condition loss
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        if use_boundary:
            if HAS_ADVANCED:
                self.boundary_loss = BoundaryConditionLoss(loss_type="huber", huber_c=huber_c)
            else:
                self.boundary_loss = BoundaryConditionLossSimple(huber_c=huber_c)

        # Structure preservation loss
        self.use_structure = use_structure
        self.structure_weight = structure_weight
        if use_structure:
            if HAS_ADVANCED:
                self.structure_loss = StructurePreservationLoss()
            else:
                self.structure_loss = StructurePreservationLossSimple()

        # Adaptive weighting
        self.use_adaptive_weighting = use_adaptive_weighting
        if use_adaptive_weighting:
            if HAS_ADVANCED:
                self.weighting = AdaptiveWeighting(
                    weighting_type=weighting_type,
                    sigma_data=sigma_data,
                )
            else:
                self.weighting = AdaptiveWeightingSimple(
                    weighting_type=weighting_type,
                    sigma_data=sigma_data,
                )

        # Feature matching (cosine similarity)
        self.feature_matching = FeatureMatchingLoss(use_cosine=True)

    def forward(
        self,
        generated: torch.Tensor,
        matched_targets: torch.Tensor,
        features_gen: Optional[torch.Tensor] = None,
        features_target: Optional[torch.Tensor] = None,
        features_memory: Optional[torch.Tensor] = None,  # Full memory features for structure loss
        coupling: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GFGW-FM loss.

        Args:
            generated: Generated images (B, C, H, W)
            matched_targets: OT-matched target images (B, C, H, W)
            features_gen: Generated sample features (B, d)
            features_target: Target sample features (B, d) - matched features
            features_memory: Full memory bank features (M, d) for structure loss
            coupling: OT coupling matrix (B, M)
            noise: Input noise (for boundary loss)
            t: Time values (for adaptive weighting)

        Returns:
            Dictionary containing individual loss terms and total loss
        """
        losses = {}
        batch_size = generated.shape[0]

        # Compute adaptive weights
        if self.use_adaptive_weighting and t is not None:
            weight = self.weighting(t)
        else:
            weight = None

        # 1. Flow matching loss (main term)
        if self.use_multiscale:
            flow_loss = self.multiscale_loss(generated, matched_targets)
        else:
            if weight is not None and hasattr(self.recon_loss, 'forward'):
                # Check if loss function supports weights
                try:
                    flow_loss = self.recon_loss(generated, matched_targets, weight)
                except TypeError:
                    flow_loss = self.recon_loss(generated, matched_targets)
            else:
                flow_loss = self.recon_loss(generated, matched_targets)

        losses['flow_loss'] = flow_loss * self.flow_loss_weight

        # 2. Feature matching loss
        if features_gen is not None and features_target is not None:
            # Use cosine similarity based feature loss
            feat_loss = self.feature_matching(features_gen, features_target)
            losses['feature_loss'] = feat_loss * self.feature_loss_weight
        else:
            losses['feature_loss'] = torch.tensor(0.0, device=generated.device)

        # 3. LPIPS perceptual loss
        if self.use_lpips and HAS_ADVANCED:
            lpips = self.lpips_loss(generated, matched_targets)
            losses['lpips_loss'] = lpips * self.lpips_weight

        # 4. Boundary condition loss
        if self.use_boundary and noise is not None and t is not None:
            boundary = self.boundary_loss(generated, noise, matched_targets, t)
            losses['boundary_loss'] = boundary * self.boundary_weight

        # 5. Structure preservation loss
        if self.use_structure and coupling is not None and features_gen is not None:
            # Use full memory features if available, otherwise fall back to matched features
            features_for_structure = features_memory if features_memory is not None else features_target
            if features_for_structure is not None:
                structure = self.structure_loss(features_gen, features_for_structure, coupling)
                losses['structure_loss'] = structure * self.structure_weight

        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses

    def to(self, device):
        """Move all components to device."""
        super().to(device)
        if self.use_lpips and HAS_ADVANCED and hasattr(self, 'lpips_loss'):
            self.lpips_loss = self.lpips_loss.to(device)
        return self


# ============================================================================
# Comprehensive Loss (from config)
# ============================================================================

class ComprehensiveFlowLoss(nn.Module):
    """
    Comprehensive flow matching loss combining all techniques.

    This is the recommended loss for GFGW-FM training with full features.
    """

    def __init__(self, config):
        """Initialize from config object."""
        super().__init__()

        self.config = config

        # Get loss config (handle both old and new config formats)
        if hasattr(config, 'loss'):
            loss_cfg = config.loss
            use_huber = loss_cfg.use_pseudo_huber
            huber_c = loss_cfg.huber_c
            use_lpips = loss_cfg.use_lpips
            lpips_weight = loss_cfg.lpips_weight
            use_boundary = loss_cfg.use_boundary_loss
            boundary_weight = loss_cfg.boundary_loss_weight
            use_structure = loss_cfg.use_structure_loss
            structure_weight = loss_cfg.structure_loss_weight
            use_adaptive = loss_cfg.use_adaptive_weighting
            weighting_type = loss_cfg.weighting_type
            flow_weight = loss_cfg.flow_loss_weight
            feature_weight = loss_cfg.feature_loss_weight
        else:
            # Fallback for old config format
            use_huber = True
            huber_c = 0.00054
            use_lpips = False
            lpips_weight = 0.5
            use_boundary = False
            boundary_weight = 0.1
            use_structure = False
            structure_weight = 0.1
            use_adaptive = True
            weighting_type = "snr"
            flow_weight = getattr(config.training, 'flow_loss_weight', 1.0)
            feature_weight = getattr(config.training, 'feature_loss_weight', 0.1)

        sigma_data = config.model.sigma_data

        # Create the main loss
        self.main_loss = GFGWFlowMatchingLoss(
            sigma_data=sigma_data,
            flow_loss_weight=flow_weight,
            feature_loss_weight=feature_weight,
            use_huber=use_huber,
            huber_c=huber_c,
            use_lpips=use_lpips,
            lpips_weight=lpips_weight,
            use_boundary=use_boundary,
            boundary_weight=boundary_weight,
            use_structure=use_structure,
            structure_weight=structure_weight,
            use_adaptive_weighting=use_adaptive,
            weighting_type=weighting_type,
            use_multiscale=True,
        )

    def forward(
        self,
        generated: torch.Tensor,
        matched_targets: torch.Tensor,
        features_gen: Optional[torch.Tensor] = None,
        features_target: Optional[torch.Tensor] = None,
        features_memory: Optional[torch.Tensor] = None,
        coupling: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with all loss components."""
        return self.main_loss(
            generated=generated,
            matched_targets=matched_targets,
            features_gen=features_gen,
            features_target=features_target,
            features_memory=features_memory,
            coupling=coupling,
            noise=noise,
            t=t,
        )

    def to(self, device):
        """Move to device."""
        super().to(device)
        self.main_loss = self.main_loss.to(device)
        return self


def create_loss_fn(config) -> ComprehensiveFlowLoss:
    """Factory function to create loss from config."""
    return ComprehensiveFlowLoss(config)
