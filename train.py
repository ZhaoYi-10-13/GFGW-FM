"""Enhanced training script for GFGW-FM.

Incorporates all advanced techniques from:
- ECM: Pseudo-Huber loss, adaptive weighting, EMA teacher
- SlimFlow: Annealing reflow, H-Flip augmentation
- Boundary RF: Boundary condition enforcement
- TCM: Two-stage training, heavy-tailed time sampling
"""

import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
import argparse

# Handle different PyTorch versions for AMP
try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = 'cuda'
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None

from configs.default import GFGWConfig, get_config, CONFIG_REGISTRY
from models.networks import OneStepGenerator, BoundaryConditionedGenerator, create_generator
from features.dino import DINOv2FeatureExtractor, GlobalFeatureMemoryBank
from losses.ot_solver import OTMatchingModule
from losses.flow_matching import GFGWFlowMatchingLoss, ComprehensiveFlowLoss, create_loss_fn
from data.dataset import ImageFolderDataset, CIFAR10Dataset, LSUNDataset, ImageNetDataset
from metrics.evaluation import FIDCalculator, TextureConsistencyScore
from utils.scheduling import TrainingScheduler, TimeSampler


class GFGWFMTrainerV2:
    """
    Enhanced trainer for GFGW-FM.

    Key improvements over V1:
    1. Two-stage training (TCM)
    2. Adaptive loss weighting (ECM)
    3. Time sampling distributions (TCM)
    4. Annealing schedules (SlimFlow)
    5. Boundary condition enforcement (Boundary RF)
    6. Mixed precision training
    7. Enhanced EMA with teacher model
    """

    def __init__(
        self,
        config: GFGWConfig,
        dataset,
        device: torch.device = torch.device("cuda"),
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset

        self._setup_logging()
        self._build_model()
        self._build_feature_extractor()
        self._build_loss()
        self._build_optimizer()
        self._build_scheduler()

        # OT solver and memory bank (initialized in train())
        self.memory_bank = None
        self.ot_solver = None
        self._image_cache = None

        # Mixed precision scaler
        self.use_amp = config.training.use_amp
        if self.use_amp:
            if AMP_DEVICE_TYPE is not None:
                self.scaler = GradScaler('cuda')
            else:
                self.scaler = GradScaler()
        else:
            self.scaler = None

        # Training state
        self.cur_nimg = 0
        self.cur_step = 0

    def _setup_logging(self):
        """Setup logging and checkpointing."""
        if self.rank == 0:
            os.makedirs(self.config.log.run_dir, exist_ok=True)
            self.stats_file = open(
                os.path.join(self.config.log.run_dir, 'stats.jsonl'), 'a'
            )

            # Save config
            config_path = os.path.join(self.config.log.run_dir, 'config.json')
            # Note: dataclass to dict conversion would go here

    def _build_model(self):
        """Build the generator model."""
        self.print0("Building generator model...")

        # Use factory function for model creation
        self.generator = create_generator(self.config).to(self.device)

        # EMA model (also as teacher for consistency loss)
        self.ema_generator = copy.deepcopy(self.generator)
        self.ema_generator.eval().requires_grad_(False)

        # Teacher model for consistency training (ECM style)
        if self.config.training.use_teacher_forcing:
            self.teacher_generator = copy.deepcopy(self.generator)
            self.teacher_generator.eval().requires_grad_(False)
        else:
            self.teacher_generator = None

        # DDP wrapper
        if self.world_size > 1:
            self.generator_ddp = nn.parallel.DistributedDataParallel(
                self.generator,
                device_ids=[self.device],
                broadcast_buffers=False
            )
        else:
            self.generator_ddp = self.generator

        num_params = sum(p.numel() for p in self.generator.parameters())
        self.print0(f"Generator parameters: {num_params:,}")

    def _build_feature_extractor(self):
        """Build DINOv2 feature extractor."""
        self.print0("Building DINOv2 feature extractor...")

        self.feature_extractor = DINOv2FeatureExtractor(
            model_name=self.config.dino.model_name,
            feature_dim=self.config.dino.feature_dim,
            normalize_features=self.config.dino.normalize_features,
            device=self.device,
        )

    def _build_loss(self):
        """Build loss functions."""
        self.print0("Building loss functions...")

        # Use comprehensive loss with all enhancements
        self.loss_fn = create_loss_fn(self.config).to(self.device)

    def _build_optimizer(self):
        """Build optimizer with optional learning rate schedule."""
        self.print0("Building optimizer...")

        if self.config.training.use_adam_w:
            self.optimizer = torch.optim.AdamW(
                self.generator.parameters(),
                lr=self.config.training.lr,
                betas=self.config.training.betas,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.generator.parameters(),
                lr=self.config.training.lr,
                betas=self.config.training.betas,
            )

    def _build_scheduler(self):
        """Build training scheduler for all dynamic parameters."""
        self.print0("Building training scheduler...")
        self.scheduler = TrainingScheduler(self.config)

    def print0(self, *args, **kwargs):
        """Print only on rank 0."""
        if self.rank == 0:
            print(*args, **kwargs)

    def _autocast(self):
        """Return autocast context manager compatible with PyTorch version."""
        if AMP_DEVICE_TYPE is not None:
            return autocast(device_type=AMP_DEVICE_TYPE, enabled=self.use_amp)
        else:
            return autocast(enabled=self.use_amp)

    def _update_ema(self, cur_nimg: int):
        """Update EMA model with adaptive decay."""
        ema_halflife_nimg = self.config.training.ema_halflife_kimg * 1000

        if self.config.training.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(
                ema_halflife_nimg,
                cur_nimg * self.config.training.ema_rampup_ratio
            )

        ema_beta = 0.5 ** (self.config.training.batch_size / max(ema_halflife_nimg, 1e-8))

        for p_ema, p_net in zip(self.ema_generator.parameters(), self.generator.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

    def _update_teacher(self, cur_step: int):
        """Update teacher model (for ECM-style consistency training)."""
        if self.teacher_generator is None:
            return

        if cur_step % self.config.training.teacher_update_interval != 0:
            return

        ema_decay = self.config.training.teacher_ema_decay

        for p_teacher, p_net in zip(self.teacher_generator.parameters(), self.generator.parameters()):
            p_teacher.copy_(p_net.detach().lerp(p_teacher, ema_decay))

    def _save_checkpoint(self, cur_kimg: int, is_latest: bool = False):
        """Save model checkpoint."""
        if self.rank != 0:
            return

        data = {
            'generator': self.generator.state_dict(),
            'ema_generator': self.ema_generator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'cur_kimg': cur_kimg,
            'cur_nimg': self.cur_nimg,
            'cur_step': self.cur_step,
        }

        if self.teacher_generator is not None:
            data['teacher_generator'] = self.teacher_generator.state_dict()

        if self.scaler is not None:
            data['scaler'] = self.scaler.state_dict()

        if is_latest:
            path = os.path.join(self.config.log.run_dir, 'checkpoint_latest.pt')
        else:
            path = os.path.join(self.config.log.run_dir, f'checkpoint_{cur_kimg:06d}.pt')

        torch.save(data, path)
        self.print0(f"Saved checkpoint to {path}")

    def _log_stats(self, stats: Dict[str, Any]):
        """Log training statistics."""
        if self.rank != 0:
            return

        self.stats_file.write(json.dumps(stats) + '\n')
        self.stats_file.flush()

    @torch.no_grad()
    def _generate_samples(self, num_samples: int = 64) -> torch.Tensor:
        """Generate sample images for visualization."""
        self.ema_generator.eval()

        z = torch.randn(
            num_samples,
            self.config.model.img_channels,
            self.config.model.img_resolution,
            self.config.model.img_resolution,
            device=self.device
        )

        if self.config.model.label_dim > 0:
            labels = torch.eye(self.config.model.label_dim, device=self.device)
            labels = labels.repeat(num_samples // self.config.model.label_dim + 1, 1)
            labels = labels[:num_samples]
        else:
            labels = None

        images = self.ema_generator(z, labels)
        return images

    def _get_images_by_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Get images from dataset by indices."""
        if hasattr(self.dataset, 'get_images_by_indices'):
            images = self.dataset.get_images_by_indices(indices)
        else:
            images = []
            for idx in indices.cpu().numpy():
                img, _, _ = self.dataset[int(idx)]
                images.append(torch.from_numpy(img))
            images = torch.stack(images, dim=0)

        images = images.float().to(self.device)
        if images.max() > 1:
            images = images / 127.5 - 1

        return images

    def _apply_augmentation(
        self,
        images: torch.Tensor,
        generated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation (H-Flip from SlimFlow)."""
        if self.config.training.use_hflip and torch.rand(1) > 0.5:
            images = torch.flip(images, dims=[-1])
            generated = torch.flip(generated, dims=[-1])

        return images, generated

    def train_step(
        self,
        real_images: torch.Tensor,
        real_labels: Optional[torch.Tensor],
        real_indices: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Perform one training step with all enhancements.

        Key steps:
        1. Sample time values using scheduler
        2. Sample z and generate x = G_θ(z)
        3. Apply augmentations
        4. Compute features and solve FGW OT
        5. Get matched targets
        6. Compute comprehensive loss
        7. Update parameters
        """
        batch_size = real_images.shape[0]
        device = real_images.device

        # Get current schedule values
        schedule_state = self.scheduler.get_state(self.cur_step)
        current_epsilon = schedule_state['epsilon']
        current_fgw_lambda = schedule_state['fgw_lambda']

        # Sample time values (using heavy-tailed distribution)
        t = self.scheduler.sample_time(batch_size, device, self.cur_step)

        # Step 1: Sample latent codes and generate images
        z = torch.randn_like(real_images)

        self.generator.train()

        # Mixed precision forward pass
        with self._autocast():
            # Generate with time conditioning if using boundary conditions
            if isinstance(self.generator, BoundaryConditionedGenerator):
                generated = self.generator(z, real_labels, t)
            else:
                generated = self.generator(z, real_labels)

        # Apply augmentations
        if self.config.training.use_hflip:
            real_images, generated = self._apply_augmentation(real_images, generated.detach())
            # Re-generate after augmentation for gradient computation
            if self.config.training.use_hflip and torch.rand(1) > 0.5:
                z = torch.flip(z, dims=[-1])
                with self._autocast():
                    if isinstance(self.generator, BoundaryConditionedGenerator):
                        generated = self.generator(z, real_labels, t)
                    else:
                        generated = self.generator(z, real_labels)

        # Step 2: Extract features
        with torch.no_grad():
            features_gen = self.feature_extractor(generated.detach())

            # Get features from memory bank
            ot_batch_size = min(self.config.ot.memory_batch_size, self.memory_bank.num_valid)
            memory_features, memory_indices, memory_distances = self.memory_bank.sample_subset(
                size=ot_batch_size,
                return_distances=True
            )

            # Compute distance matrix for generated samples
            D_gen = torch.cdist(features_gen, features_gen, p=2)

            # Step 3: Compute OT coupling using FGW with current schedule
            coupling, cost_matrix = self.ot_solver(
                features_gen=features_gen,
                features_real=memory_features,
                D_gen=D_gen,
                D_real=memory_distances,
                data_indices=memory_indices,
                epsilon=current_epsilon,
                fgw_lambda=current_fgw_lambda,
            )

            # Step 4: Get matched targets
            assignments = coupling.argmax(dim=1)
            matched_dataset_indices = memory_indices[assignments]
            matched_targets = self._get_images_by_indices(matched_dataset_indices)
            matched_features = memory_features[assignments]

        # Step 5: Re-generate with gradient and compute loss
        with self._autocast():
            if isinstance(self.generator, BoundaryConditionedGenerator):
                generated = self.generator(z, real_labels, t)
            else:
                generated = self.generator(z, real_labels)

            features_gen_grad = self.feature_extractor.forward(generated)

            # Comprehensive loss with all components
            losses = self.loss_fn(
                generated=generated,
                matched_targets=matched_targets,
                features_gen=features_gen_grad,
                features_target=matched_features,
                coupling=coupling,
                noise=z,
                t=t,
            )

        # Step 6: Backward pass with mixed precision
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(losses['total_loss']).backward()

            # Gradient clipping
            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    self.config.training.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()

            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.generator.parameters(),
                    self.config.training.grad_clip
                )

            self.optimizer.step()

        # Update step counter
        self.cur_step += 1

        return {k: v.item() for k, v in losses.items()}

    def train(self, dataloader: DataLoader):
        """Main training loop with all enhancements."""
        self.print0("=" * 60)
        self.print0("Starting GFGW-FM V2 Training")
        self.print0("=" * 60)

        # Initialize memory bank
        dataset_size = len(dataloader.dataset)
        self.print0(f"Dataset size: {dataset_size}")

        self.memory_bank = GlobalFeatureMemoryBank(
            feature_dim=self.config.dino.feature_dim,
            max_size=dataset_size,
            device=self.device,
        )

        # Pre-compute features for entire dataset
        self.print0("Pre-computing dataset features for global memory bank...")
        self.memory_bank.initialize_from_dataset(
            feature_extractor=self.feature_extractor,
            dataloader=dataloader,
        )

        # Pre-compute structure distance matrix
        self.print0("Computing structure distance matrix...")
        self.memory_bank.compute_distance_matrix(
            use_cosine=self.config.ot.use_cosine_distance
        )

        # Initialize OT solver
        self.ot_solver = OTMatchingModule(
            feature_dim=self.config.dino.feature_dim,
            num_data_points=self.memory_bank.num_valid,
            fgw_lambda=self.config.ot.fgw_lambda,
            epsilon=self.config.ot.epsilon,
            num_sinkhorn_iters=self.config.ot.num_sinkhorn_iters,
            num_fgw_iters=self.config.ot.num_fgw_iters,
            use_cosine_distance=self.config.ot.use_cosine_distance,
            dual_lr=self.config.ot.dual_lr,
            device=self.device,
        )

        self.print0("Starting training loop...")
        self.print0(f"Two-stage training: {self.config.schedule.use_two_stage}")
        self.print0(f"Time sampling: {self.config.schedule.time_sampling}")
        self.print0(f"Mixed precision: {self.use_amp}")

        # Training loop
        cur_tick = 0
        tick_start_time = time.time()

        total_kimg = self.config.training.total_kimg
        data_iter = iter(dataloader)

        while self.cur_nimg < total_kimg * 1000:
            # Get batch
            try:
                batch_data = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_data = next(data_iter)

            # Handle batch format
            if len(batch_data) == 3:
                images, labels, indices = batch_data
            else:
                images, labels = batch_data
                indices = torch.arange(images.shape[0])

            images = images.to(self.device).float()
            if images.max() > 1:
                images = images / 127.5 - 1

            if labels.ndim == 1 and self.config.model.label_dim > 0:
                labels = F.one_hot(labels, self.config.model.label_dim).float()
            labels = labels.to(self.device) if self.config.model.label_dim > 0 else None
            indices = indices.to(self.device)

            # Update learning rate
            current_lr = self.scheduler.step_optimizer(self.optimizer, self.cur_step)

            # Training step
            losses = self.train_step(images, labels, indices)

            # Update EMA and teacher
            self.cur_nimg += images.shape[0]
            self._update_ema(self.cur_nimg)
            self._update_teacher(self.cur_step)

            # Logging
            cur_kimg = self.cur_nimg // 1000

            if cur_kimg > cur_tick:
                elapsed = time.time() - tick_start_time
                schedule_state = self.scheduler.get_state(self.cur_step)

                self.print0(
                    f"kimg {cur_kimg:>6d} | "
                    f"loss {losses['total_loss']:.4f} | "
                    f"flow {losses['flow_loss']:.4f} | "
                    f"feat {losses['feature_loss']:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"eps {schedule_state['epsilon']:.3f} | "
                    f"t_range {schedule_state['time_range']} | "
                    f"sec/kimg {elapsed:.1f}"
                )

                self._log_stats({
                    'kimg': cur_kimg,
                    'step': self.cur_step,
                    'loss': losses['total_loss'],
                    'flow_loss': losses['flow_loss'],
                    'feature_loss': losses['feature_loss'],
                    'lr': current_lr,
                    'epsilon': schedule_state['epsilon'],
                    'fgw_lambda': schedule_state['fgw_lambda'],
                    'timestamp': time.time(),
                })

                cur_tick = cur_kimg
                tick_start_time = time.time()

            # Save checkpoint
            if cur_kimg > 0 and cur_kimg % self.config.log.save_every_kimg == 0:
                self._save_checkpoint(cur_kimg)
                self._save_checkpoint(cur_kimg, is_latest=True)

            # Generate samples
            if cur_kimg > 0 and cur_kimg % self.config.eval.sample_every_kimg == 0:
                if self.rank == 0:
                    samples = self._generate_samples(64)
                    save_image_grid(
                        samples,
                        os.path.join(self.config.log.run_dir, f'samples_{cur_kimg:06d}.png'),
                        drange=[-1, 1]
                    )

            # Evaluation
            if cur_kimg > 0 and cur_kimg % self.config.eval.eval_every_kimg == 0:
                self.evaluate(dataloader)

        # Final checkpoint
        self._save_checkpoint(cur_kimg, is_latest=True)
        self.print0("Training complete!")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader):
        """Run evaluation metrics."""
        self.print0("Running evaluation...")

        num_samples = min(self.config.eval.fid_num_samples, 10000)
        all_samples = []

        batch_size = self.config.training.batch_gpu
        for i in range(0, num_samples, batch_size):
            samples = self._generate_samples(min(batch_size, num_samples - i))
            all_samples.append(samples.cpu())

        all_samples = torch.cat(all_samples, dim=0)

        # Get real samples
        all_real = []
        for batch_data in dataloader:
            images = batch_data[0]
            all_real.append(images)
            if len(torch.cat(all_real, dim=0)) >= num_samples:
                break
        all_real = torch.cat(all_real, dim=0)[:num_samples]
        if all_real.max() > 1:
            all_real = all_real.float() / 127.5 - 1

        # Compute FID
        fid_calc = FIDCalculator(device=self.device)
        fid = fid_calc(all_real, all_samples)
        self.print0(f"FID: {fid:.2f}")

        # Compute TCS
        tcs_calc = TextureConsistencyScore(device=self.device)
        tcs = tcs_calc(all_real, all_samples)
        self.print0(f"TCS: {tcs:.2f}")

        return {'fid': fid, 'tcs': tcs}


def save_image_grid(images, path, drange, grid_size=None):
    """Save images as a grid."""
    import PIL.Image

    lo, hi = drange
    images = (images - lo) * (255 / (hi - lo))
    images = images.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    n = images.shape[0]
    if grid_size is None:
        gw = int(np.ceil(np.sqrt(n)))
        gh = int(np.ceil(n / gw))
    else:
        gw, gh = grid_size

    h, w, c = images.shape[1:]
    grid = np.zeros((gh * h, gw * w, c), dtype=np.uint8)

    for i, img in enumerate(images):
        y = (i // gw) * h
        x = (i % gw) * w
        grid[y:y+h, x:x+w] = img

    if c == 1:
        PIL.Image.fromarray(grid[:, :, 0], 'L').save(path)
    else:
        PIL.Image.fromarray(grid, 'RGB').save(path)


# ============================================================================
# Backward compatibility - original trainer
# ============================================================================

GFGWFMTrainer = GFGWFMTrainerV2


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='GFGW-FM V2 Training')
    parser.add_argument('--config', type=str, default='cifar10',
                        choices=list(CONFIG_REGISTRY.keys()),
                        help='Configuration preset to use')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--run-dir', type=str, default='./runs',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate from config')
    parser.add_argument('--total-kimg', type=int, default=None,
                        help='Override total training kimg')

    # New arguments for enhanced training
    parser.add_argument('--no-boundary', action='store_true',
                        help='Disable boundary condition enforcement')
    parser.add_argument('--no-two-stage', action='store_true',
                        help='Disable two-stage training')
    parser.add_argument('--no-lpips', action='store_true',
                        help='Disable LPIPS loss')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')

    args = parser.parse_args()

    # Get config from registry
    config = get_config(args.config)

    # Override config with command line args
    config.data.data_path = args.data_path
    config.log.run_dir = args.run_dir

    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.total_kimg is not None:
        config.training.total_kimg = args.total_kimg

    # Apply feature flags
    if args.no_boundary:
        config.model.use_boundary_condition = False
    if args.no_two_stage:
        config.schedule.use_two_stage = False
    if args.no_lpips:
        config.loss.use_lpips = False
    if args.no_amp:
        config.training.use_amp = False

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build dataset
    dataset_name = config.data.dataset_name
    print(f"Loading dataset: {dataset_name}")

    if 'cifar' in dataset_name:
        dataset = CIFAR10Dataset(
            root=config.data.data_path,
            train=True,
            download=True,
        )
    elif 'lsun' in dataset_name:
        category = dataset_name.replace('lsun_', '')
        dataset = LSUNDataset(
            root=config.data.data_path,
            category=category,
            split='train',
            resolution=config.data.resolution,
        )
    elif 'imagenet' in dataset_name:
        dataset = ImageNetDataset(
            root=config.data.data_path,
            split='train',
            resolution=config.data.resolution,
            use_labels=config.data.use_labels,
        )
    else:
        dataset = ImageFolderDataset(
            path=config.data.data_path,
            resolution=config.data.resolution,
            use_labels=config.data.use_labels,
        )

    print(f"Dataset size: {len(dataset)}")

    # Update label_dim
    if hasattr(dataset, 'label_dim') and dataset.label_dim != config.model.label_dim:
        print(f"Updating label_dim from {config.model.label_dim} to {dataset.label_dim}")
        config.model.label_dim = dataset.label_dim

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_gpu,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Build trainer
    trainer = GFGWFMTrainerV2(config, dataset=dataset, device=device)

    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.generator.load_state_dict(checkpoint['generator'])
        trainer.ema_generator.load_state_dict(checkpoint['ema_generator'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'cur_nimg' in checkpoint:
            trainer.cur_nimg = checkpoint['cur_nimg']
        if 'cur_step' in checkpoint:
            trainer.cur_step = checkpoint['cur_step']
        if 'teacher_generator' in checkpoint and trainer.teacher_generator is not None:
            trainer.teacher_generator.load_state_dict(checkpoint['teacher_generator'])
        if 'scaler' in checkpoint and trainer.scaler is not None:
            trainer.scaler.load_state_dict(checkpoint['scaler'])

        print(f"Resumed from {args.resume}")

    # Train
    trainer.train(dataloader)


if __name__ == '__main__':
    main()
