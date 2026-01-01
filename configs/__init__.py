"""GFGW-FM: Global Fused Gromov-Wasserstein Flow Matching for one-step generation."""

from .default import (
    GFGWConfig,
    get_cifar10_config,
    get_imagenet64_config,
    get_imagenet256_config,
)

__all__ = [
    'GFGWConfig',
    'get_cifar10_config',
    'get_imagenet64_config',
    'get_imagenet256_config',
]
