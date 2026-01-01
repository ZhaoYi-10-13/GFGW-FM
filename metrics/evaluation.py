"""Evaluation metrics for GFGW-FM including FID and TCS."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from scipy import linalg


class InceptionV3Features(nn.Module):
    """
    Inception V3 feature extractor for FID computation.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        super().__init__()
        self.device = device

        # Load pretrained InceptionV3
        import torchvision.models as models
        inception = models.inception_v3(pretrained=True, transform_input=False)

        # Extract feature layers (up to pool3)
        self.blocks = nn.ModuleList()

        # Block 1
        block1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.blocks.append(block1)

        # Block 2
        block2 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.blocks.append(block2)

        # Block 3 (Mixed layers)
        block3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        self.blocks.append(block3)

        # Block 4
        block4 = nn.Sequential(
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.blocks.append(block4)

        self.to(device)
        self.eval()

        for param in self.parameters():
            param.requires_grad = False

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for InceptionV3."""
        # Resize to 299x299
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)

        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        images = self.preprocess(images)

        for block in self.blocks:
            images = block(images)

        return images.view(images.shape[0], -1)


class FIDCalculator:
    """
    Fréchet Inception Distance (FID) calculator.
    """

    def __init__(self, device: torch.device = torch.device("cuda")):
        self.device = device
        self.feature_extractor = InceptionV3Features(device)

    @torch.no_grad()
    def compute_statistics(
        self,
        images: torch.Tensor,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance of Inception features.

        Args:
            images: Input images (N, C, H, W) in range [-1, 1]
            batch_size: Batch size for processing

        Returns:
            Tuple of (mean, covariance) as numpy arrays
        """
        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            features = self.feature_extractor(batch)
            all_features.append(features.cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)

        mu = np.mean(all_features, axis=0)
        sigma = np.cov(all_features, rowvar=False)

        return mu, sigma

    def compute_fid(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
    ) -> float:
        """
        Compute FID between two distributions.

        Args:
            mu1, sigma1: Statistics of first distribution
            mu2, sigma2: Statistics of second distribution

        Returns:
            FID score
        """
        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

        return float(fid)

    @torch.no_grad()
    def __call__(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        batch_size: int = 64,
    ) -> float:
        """
        Compute FID between real and generated images.

        Args:
            real_images: Real images (N, C, H, W)
            generated_images: Generated images (M, C, H, W)
            batch_size: Batch size for processing

        Returns:
            FID score
        """
        mu_real, sigma_real = self.compute_statistics(real_images, batch_size)
        mu_gen, sigma_gen = self.compute_statistics(generated_images, batch_size)

        return self.compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)


class TextureConsistencyScore:
    """
    Texture Consistency Score (TCS) for evaluating texture fidelity.

    TCS measures how well the generated images preserve the texture
    characteristics of real images by comparing feature statistics
    at multiple scales.
    """

    def __init__(
        self,
        feature_extractor: Optional[nn.Module] = None,
        device: torch.device = torch.device("cuda"),
        scales: List[int] = [1, 2, 4],
    ):
        """
        Initialize TCS calculator.

        Args:
            feature_extractor: Feature extractor (DINOv2 by default)
            device: Device for computation
            scales: Scales for multi-scale texture analysis
        """
        self.device = device
        self.scales = scales

        if feature_extractor is None:
            # Use DINOv2 for texture features
            self.feature_extractor = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True
            ).to(device)
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        else:
            self.feature_extractor = feature_extractor

        # For patch-level features
        self.patch_size = 14

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for feature extraction."""
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2

        # Resize to 224x224 if needed
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        return images

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features from images.

        Args:
            images: Input images (B, C, H, W)

        Returns:
            Patch features (B, num_patches, feature_dim)
        """
        images = self.preprocess(images)

        # Get patch tokens from DINOv2
        features = self.feature_extractor.forward_features(images)
        patch_tokens = features['x_norm_patchtokens']

        return patch_tokens

    @torch.no_grad()
    def compute_texture_statistics(
        self,
        images: torch.Tensor,
        batch_size: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute texture statistics from images.

        Args:
            images: Input images (N, C, H, W)
            batch_size: Batch size for processing

        Returns:
            Tuple of (mean, covariance) of patch features
        """
        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            features = self.extract_patch_features(batch)
            # Flatten batch and patch dimensions
            features = features.reshape(-1, features.shape[-1])
            all_features.append(features)

        all_features = torch.cat(all_features, dim=0)

        # Compute statistics
        mean = all_features.mean(dim=0)

        # Compute covariance efficiently
        centered = all_features - mean.unsqueeze(0)
        cov = torch.mm(centered.t(), centered) / (all_features.shape[0] - 1)

        return mean, cov

    def compute_wasserstein_distance(
        self,
        mu1: torch.Tensor,
        cov1: torch.Tensor,
        mu2: torch.Tensor,
        cov2: torch.Tensor,
    ) -> float:
        """
        Compute Wasserstein-2 distance between two Gaussians.

        Args:
            mu1, cov1: Mean and covariance of first distribution
            mu2, cov2: Mean and covariance of second distribution

        Returns:
            W2 distance
        """
        # Convert to numpy for sqrtm
        cov1_np = cov1.cpu().numpy()
        cov2_np = cov2.cpu().numpy()

        diff = (mu1 - mu2).cpu().numpy()

        # Product might be almost singular
        sqrt_cov1 = linalg.sqrtm(cov1_np)
        if not np.isfinite(sqrt_cov1).all():
            sqrt_cov1 = linalg.sqrtm(cov1_np + np.eye(cov1_np.shape[0]) * 1e-6)

        product = sqrt_cov1.dot(cov2_np).dot(sqrt_cov1)
        sqrt_product = linalg.sqrtm(product)

        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real

        w2 = np.sum(diff ** 2) + np.trace(cov1_np) + np.trace(cov2_np) - 2 * np.trace(sqrt_product)

        return float(max(0, w2))

    @torch.no_grad()
    def __call__(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        batch_size: int = 32,
    ) -> float:
        """
        Compute Texture Consistency Score.

        TCS = 100 / (1 + sqrt(W2_distance))

        Higher TCS indicates better texture consistency.

        Args:
            real_images: Real images (N, C, H, W)
            generated_images: Generated images (M, C, H, W)
            batch_size: Batch size for processing

        Returns:
            TCS score (0-100, higher is better)
        """
        # Compute statistics for real images
        mu_real, cov_real = self.compute_texture_statistics(real_images, batch_size)

        # Compute statistics for generated images
        mu_gen, cov_gen = self.compute_texture_statistics(generated_images, batch_size)

        # Compute W2 distance
        w2 = self.compute_wasserstein_distance(mu_real, cov_real, mu_gen, cov_gen)

        # Convert to TCS
        tcs = 100 / (1 + np.sqrt(w2))

        return tcs


class PrecisionRecall:
    """
    Precision and Recall metrics for generative models.
    """

    def __init__(
        self,
        k: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize Precision/Recall calculator.

        Args:
            k: Number of nearest neighbors for manifold estimation
            device: Device for computation
        """
        self.k = k
        self.device = device
        self.feature_extractor = InceptionV3Features(device)

    @torch.no_grad()
    def compute_manifold_distances(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distances to k-th nearest neighbor for manifold estimation.

        Args:
            features: Feature vectors (N, d)

        Returns:
            Distances to k-th nearest neighbor (N,)
        """
        # Compute pairwise distances
        dists = torch.cdist(features, features)

        # Set diagonal to infinity to exclude self
        dists.fill_diagonal_(float('inf'))

        # Get k-th smallest distance
        kth_dists, _ = torch.kthvalue(dists, self.k, dim=1)

        return kth_dists

    @torch.no_grad()
    def __call__(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        batch_size: int = 64,
    ) -> Tuple[float, float]:
        """
        Compute Precision and Recall.

        Args:
            real_images: Real images (N, C, H, W)
            generated_images: Generated images (M, C, H, W)
            batch_size: Batch size for feature extraction

        Returns:
            Tuple of (precision, recall)
        """
        # Extract features
        real_features = []
        for i in range(0, len(real_images), batch_size):
            batch = real_images[i:i+batch_size].to(self.device)
            features = self.feature_extractor(batch)
            real_features.append(features)
        real_features = torch.cat(real_features, dim=0)

        gen_features = []
        for i in range(0, len(generated_images), batch_size):
            batch = generated_images[i:i+batch_size].to(self.device)
            features = self.feature_extractor(batch)
            gen_features.append(features)
        gen_features = torch.cat(gen_features, dim=0)

        # Compute manifold distances
        real_kth = self.compute_manifold_distances(real_features)
        gen_kth = self.compute_manifold_distances(gen_features)

        # Compute precision: fraction of generated in real manifold
        gen_to_real = torch.cdist(gen_features, real_features)
        gen_min_dist = gen_to_real.min(dim=1)[0]
        precision = (gen_min_dist <= real_kth.max()).float().mean().item()

        # Compute recall: fraction of real in generated manifold
        real_to_gen = torch.cdist(real_features, gen_features)
        real_min_dist = real_to_gen.min(dim=1)[0]
        recall = (real_min_dist <= gen_kth.max()).float().mean().item()

        return precision, recall
