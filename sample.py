"""Sampling script for GFGW-FM."""

import os
import argparse
import torch
import numpy as np
import PIL.Image
from tqdm import tqdm

from configs.default import get_config, CONFIG_REGISTRY
from models.networks import OneStepGenerator


def save_images(images, output_dir, start_idx=0):
    """Save images to directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert from [-1, 1] to [0, 255]
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    for i, img in enumerate(images):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
            PIL.Image.fromarray(img, 'L').save(
                os.path.join(output_dir, f'{start_idx + i:06d}.png')
            )
        else:
            PIL.Image.fromarray(img, 'RGB').save(
                os.path.join(output_dir, f'{start_idx + i:06d}.png')
            )


def save_image_grid(images, path, nrow=8):
    """Save images as a grid."""
    n, c, h, w = images.shape
    ncol = nrow
    nrow = (n + ncol - 1) // ncol

    # Convert from [-1, 1] to [0, 255]
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()

    # Create grid
    grid = np.zeros((nrow * h, ncol * w, c), dtype=np.uint8)
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    if c == 1:
        PIL.Image.fromarray(grid[:, :, 0], 'L').save(path)
    else:
        PIL.Image.fromarray(grid, 'RGB').save(path)


@torch.no_grad()
def sample(
    checkpoint_path: str,
    output_dir: str,
    num_samples: int = 50000,
    batch_size: int = 64,
    seed: int = 0,
    save_grid: bool = True,
    device: str = 'cuda',
):
    """
    Generate samples from a trained GFGW-FM model.

    Args:
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save samples
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        seed: Random seed
        save_grid: Save a sample grid image
        device: Device for generation
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Build model
    generator = OneStepGenerator(
        img_resolution=config.model.img_resolution,
        img_channels=config.model.img_channels,
        label_dim=config.model.label_dim,
        sigma_data=config.model.sigma_data,
        model_channels=config.model.model_channels,
        channel_mult=config.model.channel_mult,
        num_blocks=config.model.num_blocks,
        attn_resolutions=config.model.attn_resolutions,
        dropout=0,  # No dropout during sampling
        use_fp16=config.model.use_fp16,
    ).to(device)

    # Load EMA weights
    generator.load_state_dict(checkpoint['ema_generator'])
    generator.eval()

    print(f"Generating {num_samples} samples...")

    os.makedirs(output_dir, exist_ok=True)

    # Generate samples
    total_generated = 0
    all_samples = []

    with tqdm(total=num_samples) as pbar:
        while total_generated < num_samples:
            current_batch = min(batch_size, num_samples - total_generated)

            # Sample latent codes
            z = torch.randn(
                current_batch,
                config.model.img_channels,
                config.model.img_resolution,
                config.model.img_resolution,
                device=device
            )

            # Generate class labels if needed
            if config.model.label_dim > 0:
                labels = torch.eye(config.model.label_dim, device=device)
                indices = torch.randint(0, config.model.label_dim, (current_batch,))
                labels = labels[indices]
            else:
                labels = None

            # Generate images
            images = generator(z, labels)

            # Save images
            save_images(images, output_dir, total_generated)

            # Keep some for grid
            if save_grid and len(all_samples) < 64:
                all_samples.append(images[:min(64 - len(all_samples), current_batch)])

            total_generated += current_batch
            pbar.update(current_batch)

    # Save sample grid
    if save_grid and all_samples:
        grid_samples = torch.cat(all_samples, dim=0)[:64]
        save_image_grid(
            grid_samples,
            os.path.join(output_dir, 'samples_grid.png'),
            nrow=8
        )
        print(f"Saved sample grid to {os.path.join(output_dir, 'samples_grid.png')}")

    print(f"Generated {total_generated} samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='GFGW-FM Sampling')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./samples',
                        help='Directory to save samples')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for generation')
    parser.add_argument('--no-grid', action='store_true',
                        help='Do not save sample grid')

    args = parser.parse_args()

    sample(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        save_grid=not args.no_grid,
        device=args.device,
    )


if __name__ == '__main__':
    main()
