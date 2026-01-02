"""Enhanced configuration for GFGW-FM training.

Incorporates best practices from:
- ECM (Consistency Models Made Easy)
- SlimFlow (Training Smaller One-Step Models)
- Boundary RF (Improving Rectified Flow with Boundary Conditions)
- TCM (Truncated Consistency Models)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


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

    # [NEW] Boundary condition parameterization (from Boundary RF)
    use_boundary_condition: bool = True
    boundary_type: str = "mask"  # "mask", "subtraction", or "none"

    # [NEW] Network enhancements
    use_ema_teacher: bool = True  # Use EMA model as teacher (from ECM)
    channel_mult_emb: int = 4


@dataclass
class DINOConfig:
    """DINOv2 feature extractor configuration."""
    model_name: str = "dinov2_vits14"
    feature_dim: int = 384  # ViT-S/14: 384, ViT-B/14: 768, ViT-L/14: 1024
    use_registers: bool = False
    layer_index: int = -1  # -1 for last layer
    normalize_features: bool = True

    # [NEW] Multi-scale features for texture preservation
    use_multiscale: bool = True
    multiscale_layers: Tuple[int, ...] = (-1, -4, -8)  # Multiple layers for features

    # [NEW] Feature augmentation
    feature_noise_std: float = 0.0  # Add noise to features for regularization


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

    # [NEW] Adaptive epsilon schedule (from analysis)
    use_adaptive_epsilon: bool = True
    epsilon_init: float = 0.1
    epsilon_final: float = 0.01
    epsilon_decay_kimg: int = 50

    # [NEW] Enhanced OT solving
    use_unbalanced_ot: bool = False  # Unbalanced OT for flexibility
    unbalanced_reg: float = 1.0  # KL divergence weight for unbalanced OT

    # [NEW] Coupling refinement
    coupling_temperature: float = 0.1  # Temperature for soft coupling -> hard assignment
    use_hungarian: bool = False  # Use Hungarian algorithm for exact assignment


@dataclass
class LossConfig:
    """Loss function configuration."""
    # [NEW] Flow matching loss
    flow_loss_weight: float = 1.0
    feature_loss_weight: float = 0.1

    # [NEW] Pseudo-Huber loss (from ECM/TCM)
    use_pseudo_huber: bool = True
    huber_c: float = 0.00054  # c parameter, scaled by sqrt(d) automatically

    # [NEW] Perceptual loss (from SlimFlow)
    use_lpips: bool = True
    lpips_weight: float = 0.5
    lpips_net: str = "vgg"  # "vgg" or "alex"

    # [NEW] Adaptive weighting (from ECM)
    use_adaptive_weighting: bool = True
    weighting_type: str = "snr"  # "snr", "uniform", "truncated_snr"

    # [NEW] Boundary loss (from Boundary RF / TCM)
    use_boundary_loss: bool = True
    boundary_loss_weight: float = 0.1

    # [NEW] Consistency loss (from TCM)
    use_consistency_loss: bool = True
    consistency_loss_weight: float = 0.5

    # [NEW] Multi-scale loss
    use_multiscale: bool = True

    # [NEW] Structure preservation loss
    use_structure_loss: bool = True
    structure_loss_weight: float = 0.1


@dataclass
class ScheduleConfig:
    """Training schedule configuration."""
    # [NEW] Time sampling (from TCM)
    time_sampling: str = "logit_student_t"  # "uniform", "logit_normal", "logit_student_t"
    student_t_df: float = 1.0  # Degrees of freedom for Student-t
    logit_mean: float = 0.0
    logit_std: float = 1.0

    # [NEW] Time range (from TCM - truncated training)
    use_truncated_time: bool = True
    t_min: float = 0.001
    t_max: float = 1.0

    # [NEW] Two-stage training (from TCM)
    use_two_stage: bool = True
    stage1_t_min: float = 0.7  # Stage 1: focus on boundary
    stage1_t_max: float = 1.0
    stage1_kimg: int = 20000  # Duration of stage 1

    # [NEW] Annealing schedule (from SlimFlow)
    use_annealing: bool = True
    annealing_type: str = "cosine"  # "cosine", "linear", "exponential"
    annealing_warmup_kimg: int = 50

    # [NEW] OT coupling annealing
    ot_annealing_start: float = 0.1  # Start with soft coupling
    ot_annealing_end: float = 1.0  # End with hard coupling

    # [NEW] FGW lambda annealing
    fgw_lambda_annealing: bool = True
    fgw_lambda_start: float = 0.1
    fgw_lambda_end: float = 0.5


@dataclass
class PretrainedConfig:
    """Pretrained model configuration (from ECM/TCM best practices)."""
    # Pretrained model loading
    use_pretrained: bool = True  # Enable pretrained initialization
    pretrained_path: Optional[str] = None  # Path to pretrained .pkl or .pt file
    pretrained_url: Optional[str] = None  # URL for downloading pretrained model
    pretrained_strict: bool = False  # Whether to require exact match
    
    # Supported pretrained models (from EDM/TCM)
    # CIFAR-10: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl
    # CIFAR-10 Cond: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    # ImageNet-64: edm2-img64-s-1073741-0.130.pkl (from EDM2)
    # FFHQ-64: https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl
    
    # Model format handling
    model_format: str = "edm"  # "edm", "edm2", "ect", "tcm"
    ema_key: str = "ema"  # Key for EMA weights in checkpoint
    
    # Transfer learning options
    freeze_encoder: bool = False  # Freeze encoder weights initially
    freeze_encoder_kimg: int = 0  # Unfreeze after this many kimg
    
    # Architecture adaptation
    adapt_architecture: bool = True  # Adapt if architecture slightly differs


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

    # [NEW] Learning rate schedule
    use_lr_schedule: bool = True
    lr_schedule: str = "cosine"  # "cosine", "constant", "linear_decay"
    lr_warmup_kimg: int = 10
    lr_min_ratio: float = 0.01

    # [NEW] Optimizer enhancements
    use_adam_w: bool = True
    use_gradient_accumulation: bool = False
    gradient_accumulation_steps: int = 1

    # [NEW] Mixed precision
    use_amp: bool = True  # Automatic mixed precision

    # [NEW] Data augmentation (from SlimFlow)
    use_hflip: bool = True
    use_color_jitter: bool = False
    color_jitter_strength: float = 0.1

    # [NEW] Regularization
    use_r1_reg: bool = False  # R1 regularization
    r1_gamma: float = 10.0
    r1_interval: int = 16

    # [NEW] Teacher model (from ECM style training)
    use_teacher_forcing: bool = True
    teacher_ema_decay: float = 0.9999
    teacher_update_interval: int = 1
    
    # [NEW] Learning rate for fine-tuning pretrained models
    # (smaller LR recommended when using pretrained initialization)
    finetune_lr_multiplier: float = 0.1  # LR = lr * multiplier when finetuning


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

    # [NEW] Data preprocessing
    normalize_type: str = "symmetric"  # "symmetric" [-1,1], "standard" [0,1]

    # [NEW] Dynamic dataset loading
    use_streaming: bool = False  # For large datasets


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    fid_num_samples: int = 50000
    tcs_num_samples: int = 10000
    eval_every_kimg: int = 10000
    sample_every_kimg: int = 1000

    # [NEW] Enhanced evaluation
    compute_kid: bool = True  # Kernel Inception Distance
    compute_precision_recall: bool = True
    compute_is: bool = False  # Inception Score

    # [NEW] NFE evaluation
    eval_nfe_list: Tuple[int, ...] = (1, 2, 4)  # Evaluate with different NFE


@dataclass
class LogConfig:
    """Logging configuration."""
    run_dir: str = "./runs"
    log_every: int = 100
    save_every_kimg: int = 10000
    keep_checkpoints: int = 5

    # [NEW] Enhanced logging
    use_wandb: bool = False
    wandb_project: str = "gfgw-fm"
    log_images: bool = True
    log_images_every_kimg: int = 1000
    num_log_images: int = 64

    # [NEW] Trajectory visualization
    log_trajectories: bool = True
    trajectory_log_interval: int = 5000


@dataclass
class GFGWConfig:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    ot: OTConfig = field(default_factory=OTConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pretrained: PretrainedConfig = field(default_factory=PretrainedConfig)  # [NEW] Pretrained model config
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    log: LogConfig = field(default_factory=LogConfig)

    seed: int = 0
    device: str = "cuda"
    distributed: bool = True

    # [NEW] Experiment tracking
    experiment_name: str = "gfgw_fm"
    experiment_version: str = "v2.0"


def get_cifar10_config() -> GFGWConfig:
    """Get configuration for CIFAR-10."""
    config = GFGWConfig()
    config.model.img_resolution = 32
    config.model.img_channels = 3
    config.model.label_dim = 10
    config.data.dataset_name = "cifar10"
    config.data.resolution = 32

    # Optimized settings for CIFAR-10
    config.loss.huber_c = 0.00054 * np.sqrt(3 * 32 * 32)
    config.schedule.stage1_kimg = 10000
    config.training.total_kimg = 50000
    
    # [NEW] Pretrained model configuration (from EDM - same as ECM/TCM)
    config.pretrained.use_pretrained = True
    config.pretrained.pretrained_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl"
    config.pretrained.model_format = "edm"
    
    # Use smaller LR for fine-tuning (like ECM "1 GPU hour" setting)
    config.training.lr = 1e-4
    config.training.finetune_lr_multiplier = 1.0  # Already lower

    return config


def get_cifar10_cond_config() -> GFGWConfig:
    """Get configuration for CIFAR-10 conditional."""
    config = get_cifar10_config()
    config.pretrained.pretrained_url = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl"
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

    # Optimized settings for ImageNet-64
    config.loss.huber_c = 0.00054 * np.sqrt(3 * 64 * 64)
    config.schedule.stage1_kimg = 30000
    config.training.total_kimg = 200000
    
    # [NEW] Pretrained model configuration (from EDM2 - same as TCM)
    config.pretrained.use_pretrained = True
    # EDM2 ImageNet-64 model (S size, 280M params)
    config.pretrained.pretrained_url = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.130.pkl"
    config.pretrained.model_format = "edm2"
    
    # Use smaller LR for fine-tuning
    config.training.lr = 5e-5
    
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

    # Optimized settings for ImageNet-256
    config.loss.huber_c = 0.00054 * np.sqrt(3 * 256 * 256)
    config.loss.use_lpips = True
    config.loss.lpips_weight = 1.0
    config.schedule.stage1_kimg = 50000
    config.training.total_kimg = 400000

    return config


def get_lsun_bedroom_config() -> GFGWConfig:
    """Get configuration for LSUN Bedroom 256x256."""
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
    config.training.total_kimg = 200000
    config.dino.model_name = "dinov2_vitl14"
    config.dino.feature_dim = 1024

    # Optimized settings
    config.loss.huber_c = 0.00054 * np.sqrt(3 * 256 * 256)
    config.schedule.stage1_kimg = 40000

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
    'cifar10_cond': get_cifar10_cond_config,
    'imagenet64': get_imagenet64_config,
    'imagenet256': get_imagenet256_config,
    'lsun_bedroom': get_lsun_bedroom_config,
    'lsun_church': get_lsun_church_config,
    'lsun_cat': get_lsun_cat_config,
}


# Pretrained model URLs registry
PRETRAINED_URLS = {
    # EDM pretrained models (CIFAR-10)
    'edm-cifar10-uncond': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl',
    'edm-cifar10-cond': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl',
    
    # EDM pretrained models (FFHQ)
    'edm-ffhq64-uncond': 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl',
    
    # EDM2 pretrained models (ImageNet-64)
    'edm2-img64-s': 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.130.pkl',
    'edm2-img64-m': 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-m-2147483-0.060.pkl',
    'edm2-img64-l': 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-l-1073741-0.030.pkl',
    'edm2-img64-xl': 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xl-536870-0.015.pkl',
}


def get_config(name: str) -> GFGWConfig:
    """Get configuration by name."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[name]()
