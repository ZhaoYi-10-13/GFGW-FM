"""Evaluation script for GFGW-FM."""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from configs.default import get_config, CONFIG_REGISTRY
from models.networks import OneStepGenerator
from data.dataset import ImageFolderDataset, CIFAR10Dataset, LSUNDataset, ImageNetDataset
from metrics.evaluation import FIDCalculator, TextureConsistencyScore, PrecisionRecall


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_path: str,
    num_samples: int = 50000,
    batch_size: int = 64,
    device: str = 'cuda',
):
    """
    Evaluate a trained GFGW-FM model.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to real data
        num_samples: Number of samples for evaluation
        batch_size: Batch size for generation
        device: Device for evaluation
    """
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
        dropout=0,
        use_fp16=config.model.use_fp16,
    ).to(device)

    generator.load_state_dict(checkpoint['ema_generator'])
    generator.eval()

    # Load real data
    print("Loading real data...")
    if 'cifar' in data_path.lower():
        dataset = CIFAR10Dataset(root=data_path, train=True)
    else:
        dataset = ImageFolderDataset(path=data_path, resolution=config.model.img_resolution)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    # Generate samples
    print(f"Generating {num_samples} samples...")
    all_fake = []
    all_real = []

    # Generate fake samples
    num_generated = 0
    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)

        z = torch.randn(
            current_batch,
            config.model.img_channels,
            config.model.img_resolution,
            config.model.img_resolution,
            device=device
        )

        if config.model.label_dim > 0:
            labels = torch.eye(config.model.label_dim, device=device)
            indices = torch.randint(0, config.model.label_dim, (current_batch,))
            labels = labels[indices]
        else:
            labels = None

        images = generator(z, labels)
        all_fake.append(images.cpu())
        num_generated += current_batch

    all_fake = torch.cat(all_fake, dim=0)[:num_samples]

    # Load real samples
    print("Loading real samples...")
    for images, _ in tqdm(dataloader):
        if images.max() > 1:
            images = images.float() / 127.5 - 1
        all_real.append(images)
        if len(torch.cat(all_real, dim=0)) >= num_samples:
            break

    all_real = torch.cat(all_real, dim=0)[:num_samples]

    # Compute metrics
    print("Computing FID...")
    fid_calc = FIDCalculator(device=device)
    fid = fid_calc(all_real, all_fake, batch_size=batch_size)
    print(f"FID: {fid:.4f}")

    print("Computing TCS...")
    tcs_calc = TextureConsistencyScore(device=device)
    tcs = tcs_calc(all_real, all_fake, batch_size=batch_size)
    print(f"TCS: {tcs:.4f}")

    print("Computing Precision/Recall...")
    pr_calc = PrecisionRecall(device=device)
    precision, recall = pr_calc(all_real, all_fake, batch_size=batch_size)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    print(f"FID:       {fid:.4f}")
    print(f"TCS:       {tcs:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("=" * 50)

    return {
        'fid': fid,
        'tcs': tcs,
        'precision': precision,
        'recall': recall,
    }


def main():
    parser = argparse.ArgumentParser(description='GFGW-FM Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to real data')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='Number of samples for evaluation')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for evaluation')

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == '__main__':
    main()
