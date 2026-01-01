"""Advanced loss functions for GFGW-FM.

Incorporates state-of-the-art techniques from:
- ECM: Pseudo-Huber loss, adaptive weighting
- SlimFlow: LPIPS perceptual loss
- Boundary RF: Boundary condition loss
- TCM: Consistency loss, truncated weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple, Callable

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: lpips not installed. LPIPS loss will be disabled.")


class PseudoHuberLoss(nn.Module):
    """
    Pseudo-Huber loss from ECM/TCM papers.

    More robust to outliers than L2, smoother than L1.
    Loss = sqrt(||x - y||^2 + c^2) - c

    When c -> 0: approaches L1 loss
    When c -> inf: approaches L2 loss
    """

    def __init__(
        self,
        c: float = 0.00054,
        reduction: str = "mean",
        scale_by_dim: bool = True,
    ):
        super().__init__()
        self.c = c
        self.reduction = reduction
        self.scale_by_dim = scale_by_dim

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Pseudo-Huber loss.

        Args:
            pred: Predicted tensor (B, C, H, W) or (B, D)
            target: Target tensor, same shape as pred
            weight: Optional per-sample weights (B,)

        Returns:
            Scalar loss value
        """
        diff = pred - target

        # Compute squared L2 distance per sample
        if diff.dim() > 2:
            l2_sq = (diff ** 2).flatten(1).sum(dim=1)  # (B,)
        else:
            l2_sq = (diff ** 2).sum(dim=1)  # (B,)

        # Scale c by dimension if requested
        c = self.c
        if self.scale_by_dim:
            dim = diff[0].numel()
            c = self.c * np.sqrt(dim)

        # Pseudo-Huber formula
        if c > 0:
            loss = torch.sqrt(l2_sq + c ** 2) - c
        else:
            loss = torch.sqrt(l2_sq)

        # Apply weights if provided
        if weight is not None:
            loss = loss * weight

        # Reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.

    From SlimFlow - helps preserve perceptual quality and texture.
    """

    def __init__(
        self,
        net: str = "vgg",
        spatial: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        if not HAS_LPIPS:
            raise ImportError("lpips not installed. Install with: pip install lpips")
        self.lpips_fn = lpips.LPIPS(net=net, spatial=spatial)
        self.lpips_fn.requires_grad_(False)
        self.normalize = normalize

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS loss.

        Args:
            pred: Predicted images (B, C, H, W) in range [-1, 1]
            target: Target images (B, C, H, W) in range [-1, 1]

        Returns:
            Scalar LPIPS loss
        """
        # LPIPS expects inputs in [-1, 1]
        if self.normalize:
            pred = pred.clamp(-1, 1)
            target = target.clamp(-1, 1)

        loss = self.lpips_fn(pred, target)
        return loss.mean()

    def to(self, device):
        """Move LPIPS network to device."""
        super().to(device)
        self.lpips_fn = self.lpips_fn.to(device)
        return self


class BoundaryConditionLoss(nn.Module):
    """
    Boundary condition loss from Boundary RF and TCM.

    Enforces v(x, 1) = x (identity at t=1) and v(x, 0) = target.
    This helps straighten trajectories for one-step generation.
    """

    def __init__(
        self,
        boundary_type: str = "both",  # "start", "end", "both"
        loss_type: str = "l2",  # "l2", "huber"
        huber_c: float = 0.00054,
    ):
        super().__init__()
        self.boundary_type = boundary_type
        self.loss_type = loss_type
        if loss_type == "huber":
            self.loss_fn = PseudoHuberLoss(c=huber_c)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self,
        model_output: torch.Tensor,
        noise: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary condition loss.

        Args:
            model_output: Model's predicted output
            noise: Input noise z
            target: Target data x
            t: Time values (B,)

        Returns:
            Boundary loss
        """
        loss = 0.0

        # End boundary: at t=1, output should equal noise (no change)
        if self.boundary_type in ["end", "both"]:
            # Mask for samples near t=1
            end_mask = (t > 0.99).float()
            if end_mask.sum() > 0:
                end_loss = self.loss_fn(
                    model_output * end_mask.view(-1, 1, 1, 1),
                    noise * end_mask.view(-1, 1, 1, 1)
                )
                loss = loss + end_loss

        # Start boundary: at t=0, output should equal target
        if self.boundary_type in ["start", "both"]:
            # Mask for samples near t=0
            start_mask = (t < 0.01).float()
            if start_mask.sum() > 0:
                start_loss = self.loss_fn(
                    model_output * start_mask.view(-1, 1, 1, 1),
                    target * start_mask.view(-1, 1, 1, 1)
                )
                loss = loss + start_loss

        return loss


class ConsistencyLoss(nn.Module):
    """
    Consistency loss from TCM (Truncated Consistency Models).

    Enforces that the model output is consistent across different
    time steps along the same trajectory.

    f(x_t, t) should map to same x_0 for all t along trajectory.
    """

    def __init__(
        self,
        loss_type: str = "huber",
        huber_c: float = 0.00054,
        use_ema_target: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.use_ema_target = use_ema_target
        if loss_type == "huber":
            self.loss_fn = PseudoHuberLoss(c=huber_c)
        else:
            self.loss_fn = nn.MSELoss()

    def forward(
        self,
        output_t: torch.Tensor,
        output_s: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute consistency loss between two time points.

        Args:
            output_t: Model output at time t (from online model)
            output_s: Model output at time s (from target/EMA model)
            t: Time values for output_t
            s: Time values for output_s (s < t typically)

        Returns:
            Consistency loss
        """
        # Weight by time difference (larger weight for closer times)
        delta_t = torch.abs(t - s)
        weight = 1.0 / (delta_t + 0.001)
        weight = weight / weight.mean()  # Normalize

        return self.loss_fn(output_t, output_s.detach(), weight=weight)


class AdaptiveWeighting(nn.Module):
    """
    Adaptive loss weighting from ECM.

    Computes signal-to-noise ratio (SNR) based weights to balance
    loss across different noise levels.
    """

    def __init__(
        self,
        weighting_type: str = "snr",  # "snr", "uniform", "truncated_snr", "min_snr"
        sigma_data: float = 0.5,
        min_snr_gamma: float = 5.0,  # For min_snr weighting
    ):
        super().__init__()
        self.weighting_type = weighting_type
        self.sigma_data = sigma_data
        self.min_snr_gamma = min_snr_gamma

    def forward(
        self,
        t: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adaptive weights based on time/noise level.

        Args:
            t: Time values (B,)
            sigma: Optional noise levels (B,)

        Returns:
            Per-sample weights (B,)
        """
        if self.weighting_type == "uniform":
            return torch.ones_like(t)

        # Compute sigma from t if not provided
        if sigma is None:
            # Assuming linear interpolation schedule
            sigma = t * self.sigma_data

        # SNR = sigma_data^2 / sigma^2
        snr = (self.sigma_data ** 2) / (sigma ** 2 + 1e-8)

        if self.weighting_type == "snr":
            # Weight = 1 / (1 + 1/SNR) = SNR / (SNR + 1)
            weight = snr / (snr + 1)

        elif self.weighting_type == "truncated_snr":
            # Truncated SNR weighting from TCM
            # Avoid extreme weights at boundaries
            weight = torch.clamp(snr / (snr + 1), min=0.1, max=10.0)

        elif self.weighting_type == "min_snr":
            # Min-SNR weighting from "Efficient Diffusion Training"
            # weight = min(SNR, gamma) / SNR
            weight = torch.minimum(snr, torch.tensor(self.min_snr_gamma)) / snr

        else:
            weight = torch.ones_like(t)

        # Normalize weights
        weight = weight / weight.mean()

        return weight


class StructurePreservationLoss(nn.Module):
    """
    Structure preservation loss for FGW alignment.

    Ensures generated samples preserve pairwise relationships
    similar to real data distribution.
    """

    def __init__(
        self,
        distance_type: str = "l2",
        normalize: bool = True,
    ):
        super().__init__()
        self.distance_type = distance_type
        self.normalize = normalize

    def forward(
        self,
        features_gen: torch.Tensor,
        features_real: torch.Tensor,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute structure preservation loss.

        Args:
            features_gen: Generated features (n, d)
            features_real: Real features (m, d)
            coupling: OT coupling matrix (n, m)

        Returns:
            Structure preservation loss
        """
        # Compute pairwise distances
        D_gen = torch.cdist(features_gen, features_gen, p=2)
        D_real = torch.cdist(features_real, features_real, p=2)

        if self.normalize:
            D_gen = D_gen / (D_gen.max() + 1e-8)
            D_real = D_real / (D_real.max() + 1e-8)

        # Compute expected distance matrix under coupling
        # E[D_real | coupling] = coupling @ D_real @ coupling.T
        D_real_transported = torch.mm(torch.mm(coupling, D_real), coupling.T)

        # Loss is difference between generated and transported real distances
        loss = F.mse_loss(D_gen, D_real_transported)

        return loss


class FlowVelocityLoss(nn.Module):
    """
    Flow velocity matching loss with enhancements.

    Combines multiple loss components for robust training.
    """

    def __init__(
        self,
        use_pseudo_huber: bool = True,
        huber_c: float = 0.00054,
        use_lpips: bool = False,
        lpips_weight: float = 0.5,
        use_boundary: bool = True,
        boundary_weight: float = 0.1,
    ):
        super().__init__()

        # Primary reconstruction loss
        if use_pseudo_huber:
            self.recon_loss = PseudoHuberLoss(c=huber_c)
        else:
            self.recon_loss = nn.MSELoss()

        # LPIPS perceptual loss
        self.use_lpips = use_lpips
        self.lpips_weight = lpips_weight
        if use_lpips:
            self.lpips_loss = LPIPSLoss(net="vgg")

        # Boundary condition loss
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        if use_boundary:
            self.boundary_loss = BoundaryConditionLoss(loss_type="huber", huber_c=huber_c)

    def forward(
        self,
        generated: torch.Tensor,
        target: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined flow velocity loss.

        Args:
            generated: Generated images (B, C, H, W)
            target: Target images (B, C, H, W)
            noise: Input noise (B, C, H, W)
            t: Time values (B,)
            weight: Optional per-sample weights (B,)

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # Reconstruction loss
        recon = self.recon_loss(generated, target, weight)
        losses['recon_loss'] = recon

        # LPIPS loss
        if self.use_lpips:
            lpips = self.lpips_loss(generated, target)
            losses['lpips_loss'] = lpips * self.lpips_weight

        # Boundary loss
        if self.use_boundary and noise is not None and t is not None:
            boundary = self.boundary_loss(generated, noise, target, t)
            losses['boundary_loss'] = boundary * self.boundary_weight

        # Total loss
        total = sum(losses.values())
        losses['total_loss'] = total

        return losses

    def to(self, device):
        """Move all loss modules to device."""
        super().to(device)
        if self.use_lpips:
            self.lpips_loss = self.lpips_loss.to(device)
        return self


class GFGWFlowMatchingLossV2(nn.Module):
    """
    Enhanced GFGW Flow Matching Loss V2.

    Comprehensive loss combining all advanced techniques:
    - Pseudo-Huber loss for robustness
    - LPIPS for perceptual quality
    - Boundary conditions for trajectory straightening
    - Consistency for temporal coherence
    - Structure preservation for FGW alignment
    - Adaptive weighting for balanced training
    """

    def __init__(
        self,
        sigma_data: float = 0.5,
        # Loss weights
        flow_loss_weight: float = 1.0,
        feature_loss_weight: float = 0.1,
        lpips_weight: float = 0.5,
        boundary_weight: float = 0.1,
        consistency_weight: float = 0.5,
        structure_weight: float = 0.1,
        # Loss options
        use_pseudo_huber: bool = True,
        huber_c: float = 0.00054,
        use_lpips: bool = True,
        use_boundary: bool = True,
        use_consistency: bool = True,
        use_structure: bool = True,
        use_adaptive_weighting: bool = True,
        weighting_type: str = "snr",
    ):
        super().__init__()

        self.sigma_data = sigma_data

        # Weights
        self.flow_loss_weight = flow_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.lpips_weight = lpips_weight
        self.boundary_weight = boundary_weight
        self.consistency_weight = consistency_weight
        self.structure_weight = structure_weight

        # Flags
        self.use_lpips = use_lpips
        self.use_boundary = use_boundary
        self.use_consistency = use_consistency
        self.use_structure = use_structure
        self.use_adaptive_weighting = use_adaptive_weighting

        # Loss components
        if use_pseudo_huber:
            self.flow_loss_fn = PseudoHuberLoss(c=huber_c)
            self.feature_loss_fn = PseudoHuberLoss(c=huber_c * 0.1)  # Smaller c for features
        else:
            self.flow_loss_fn = nn.MSELoss(reduction='none')
            self.feature_loss_fn = nn.MSELoss(reduction='none')

        if use_lpips:
            self.lpips_fn = LPIPSLoss(net="vgg")

        if use_boundary:
            self.boundary_fn = BoundaryConditionLoss(loss_type="huber", huber_c=huber_c)

        if use_consistency:
            self.consistency_fn = ConsistencyLoss(loss_type="huber", huber_c=huber_c)

        if use_structure:
            self.structure_fn = StructurePreservationLoss()

        if use_adaptive_weighting:
            self.weighting_fn = AdaptiveWeighting(
                weighting_type=weighting_type,
                sigma_data=sigma_data
            )

    def forward(
        self,
        generated: torch.Tensor,
        matched_targets: torch.Tensor,
        features_gen: torch.Tensor,
        features_target: torch.Tensor,
        coupling: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        ema_output: Optional[torch.Tensor] = None,
        t_ema: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive GFGW flow matching loss.

        Args:
            generated: Generated images (B, C, H, W)
            matched_targets: OT-matched target images (B, C, H, W)
            features_gen: Generated features (B, D)
            features_target: Target features (B, D)
            coupling: OT coupling matrix
            noise: Input noise (for boundary loss)
            t: Time values (for adaptive weighting)
            ema_output: EMA model output (for consistency loss)
            t_ema: Time for EMA output

        Returns:
            Dictionary of loss components
        """
        losses = {}
        batch_size = generated.shape[0]

        # Compute adaptive weights
        if self.use_adaptive_weighting and t is not None:
            weight = self.weighting_fn(t)
        else:
            weight = torch.ones(batch_size, device=generated.device)

        # 1. Flow matching loss (image space)
        flow_loss = self.flow_loss_fn(generated, matched_targets, weight)
        losses['flow_loss'] = flow_loss * self.flow_loss_weight

        # 2. Feature matching loss (DINOv2 space)
        feature_loss = self.feature_loss_fn(features_gen, features_target, weight)
        losses['feature_loss'] = feature_loss * self.feature_loss_weight

        # 3. LPIPS perceptual loss
        if self.use_lpips:
            lpips_loss = self.lpips_fn(generated, matched_targets)
            losses['lpips_loss'] = lpips_loss * self.lpips_weight

        # 4. Boundary condition loss
        if self.use_boundary and noise is not None and t is not None:
            boundary_loss = self.boundary_fn(generated, noise, matched_targets, t)
            losses['boundary_loss'] = boundary_loss * self.boundary_weight

        # 5. Consistency loss (with EMA model)
        if self.use_consistency and ema_output is not None and t_ema is not None and t is not None:
            consistency_loss = self.consistency_fn(generated, ema_output, t, t_ema)
            losses['consistency_loss'] = consistency_loss * self.consistency_weight

        # 6. Structure preservation loss
        if self.use_structure:
            structure_loss = self.structure_fn(features_gen, features_target, coupling)
            losses['structure_loss'] = structure_loss * self.structure_weight

        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss

        return losses

    def to(self, device):
        """Move all components to device."""
        super().to(device)
        if self.use_lpips:
            self.lpips_fn = self.lpips_fn.to(device)
        return self


def get_loss_fn(config) -> GFGWFlowMatchingLossV2:
    """Factory function to create loss from config."""
    return GFGWFlowMatchingLossV2(
        sigma_data=config.model.sigma_data,
        flow_loss_weight=config.loss.flow_loss_weight,
        feature_loss_weight=config.loss.feature_loss_weight,
        lpips_weight=config.loss.lpips_weight,
        boundary_weight=config.loss.boundary_loss_weight,
        consistency_weight=config.loss.consistency_loss_weight,
        structure_weight=config.loss.structure_loss_weight,
        use_pseudo_huber=config.loss.use_pseudo_huber,
        huber_c=config.loss.huber_c,
        use_lpips=config.loss.use_lpips,
        use_boundary=config.loss.use_boundary_loss,
        use_consistency=config.loss.use_consistency_loss,
        use_structure=config.loss.use_structure_loss,
        use_adaptive_weighting=config.loss.use_adaptive_weighting,
        weighting_type=config.loss.weighting_type,
    )
