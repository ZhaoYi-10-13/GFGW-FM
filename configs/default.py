"""Default configuration for GFGW-FM training."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    img_resolution: int = 32
    img_channels: int = 3
    model_channels: int = 128
    channel_mult: Tuple[int, ...] = (1, 2, 2, 2)
    num_blocks: int = 4
    attn_resolutions: Tuple[int, ...] = (16,)
    dropout: float = 0.1
    label_dim: int = 0
    use_fp16: bool = False
    sigma_data: float = 0.5


@dataclass
class DINOConfig:
    """DINOv2 feature extractor configuration."""
    model_name: str = "dinov2_vits14"
    feature_dim: int = 384  # ViT-S/14: 384, ViT-B/14: 768, ViT-L/14: 1024
    use_registers: bool = False
    layer_index: int = -1  # -1 for last layer
    normalize_features: bool = True


@dataclass
class OTConfig:
    """Optimal Transport solver configuration."""
    fgw_lambda: float = 0.5  # Weight for structure term (0=pure Wasserstein, 1=pure GW)
    epsilon: float = 0.05  # Entropic regularization
    num_sinkhorn_iters: int = 50
    num_fgw_iters: int = 10  # Iterations for FGW solver
    use_cosine_distance: bool = False  # Use cosine distance instead of L2
    memory_batch_size: int = 2048  # Batch size for OT computation
    dual_lr: float = 0.1  # Learning rate for dual potential updates


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    batch_gpu: int = 32
    total_kimg: int = 100000
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    ema_halflife_kimg: int = 500
    ema_rampup_ratio: float = 0.05
    grad_clip: float = 1.0

    # Loss weights
    flow_loss_weight: float = 1.0
    feature_loss_weight: float = 0.1

    # Augmentation
    use_augment: bool = True
    augment_p: float = 0.12


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "cifar10"
    data_path: str = "./data"
    resolution: int = 32
    num_workers: int = 4
    use_labels: bool = True
    xflip: bool = True
    cache: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    fid_num_samples: int = 50000
    tcs_num_samples: int = 10000
    eval_every_kimg: int = 10000
    sample_every_kimg: int = 1000


@dataclass
class LogConfig:
    """Logging configuration."""
    run_dir: str = "./runs"
    log_every: int = 100
    save_every_kimg: int = 10000
    keep_checkpoints: int = 5


@dataclass
class GFGWConfig:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)

    seed: int = 0
    device: str = "cuda"
    distributed: bool = True


def get_cifar10_config() -> GFGWConfig:
    """Get configuration for CIFAR-10."""
    config = GFGWConfig()
    config.model.img_resolution = 32
    config.model.img_channels = 3
    config.model.label_dim = 10
    config.data.dataset_name = "cifar10"
    config.data.resolution = 32
    return config


def get_imagenet64_config() -> GFGWConfig:
    """Get configuration for ImageNet 64x64."""
    config = GFGWConfig()
    config.model.img_resolution = 64
    config.model.img_channels = 3
    config.model.model_channels = 192
    config.model.channel_mult = (1, 2, 3, 4)
    config.model.label_dim = 1000
    config.data.dataset_name = "imagenet64"
    config.data.resolution = 64
    config.training.batch_size = 256
    config.dino.model_name = "dinov2_vitb14"
    config.dino.feature_dim = 768
    return config


def get_imagenet256_config() -> GFGWConfig:
    """Get configuration for ImageNet 256x256."""
    config = GFGWConfig()
    config.model.img_resolution = 256
    config.model.img_channels = 3
    config.model.model_channels = 256
    config.model.channel_mult = (1, 1, 2, 2, 4, 4)
    config.model.attn_resolutions = (32, 16, 8)
    config.model.label_dim = 1000
    config.data.dataset_name = "imagenet256"
    config.data.resolution = 256
    config.training.batch_size = 256
    config.training.batch_gpu = 8
    config.dino.model_name = "dinov2_vitl14"
    config.dino.feature_dim = 1024
    return config


def get_lsun_bedroom_config() -> GFGWConfig:
    """Get configuration for LSUN Bedroom 256x256 (used in ECM, TCM)."""
    config = GFGWConfig()
    config.model.img_resolution = 256
    config.model.img_channels = 3
    config.model.model_channels = 256
    config.model.channel_mult = (1, 1, 2, 2, 4, 4)
    config.model.attn_resolutions = (32, 16, 8)
    config.model.label_dim = 0  # Unconditional
    config.model.dropout = 0.1
    config.data.dataset_name = "lsun_bedroom"
    config.data.resolution = 256
    config.data.use_labels = False
    config.training.batch_size = 256
    config.training.batch_gpu = 8
    config.training.total_kimg = 200000  # ECM uses 200M images
    config.dino.model_name = "dinov2_vitl14"
    config.dino.feature_dim = 1024
    return config


def get_lsun_church_config() -> GFGWConfig:
    """Get configuration for LSUN Church 256x256."""
    config = get_lsun_bedroom_config()
    config.data.dataset_name = "lsun_church"
    return config


def get_lsun_cat_config() -> GFGWConfig:
    """Get configuration for LSUN Cat 256x256."""
    config = get_lsun_bedroom_config()
    config.data.dataset_name = "lsun_cat"
    return config


# Configuration registry for easy access
CONFIG_REGISTRY = {
    'cifar10': get_cifar10_config,
    'imagenet64': get_imagenet64_config,
    'imagenet256': get_imagenet256_config,
    'lsun_bedroom': get_lsun_bedroom_config,
    'lsun_church': get_lsun_church_config,
    'lsun_cat': get_lsun_cat_config,
}


def get_config(name: str) -> GFGWConfig:
    """Get configuration by name."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[name]()
