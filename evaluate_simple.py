"""Simple evaluation script for GFGW-FM without requiring config in checkpoint."""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm

from configs.default import get_cifar10_config
from models.networks import OneStepGenerator
from data.dataset import CIFAR10Dataset
from metrics.evaluation import FIDCalculator


@torch.no_grad()
def evaluate(
    checkpoint_path: str,
    data_path: str,
    num_samples: int = 50000,
    batch_size: int = 128,
    device: str = 'cuda',
):
    """Evaluate a trained GFGW-FM model."""
    # Get CIFAR-10 config
    config = get_cifar10_config()
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Build model
    print("Building generator...")
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
    
    # Load EMA weights
    generator.load_state_dict(checkpoint['ema_generator'])
    generator.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in generator.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load real data
    print("Loading CIFAR-10 dataset...")
    dataset = CIFAR10Dataset(root=data_path, train=True, download=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    all_fake = []
    
    num_generated = 0
    pbar = tqdm(total=num_samples)
    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)
        
        z = torch.randn(
            current_batch,
            config.model.img_channels,
            config.model.img_resolution,
            config.model.img_resolution,
            device=device
        )
        
        # Sample random class labels
        if config.model.label_dim > 0:
            labels = torch.eye(config.model.label_dim, device=device)
            indices = torch.randint(0, config.model.label_dim, (current_batch,))
            labels = labels[indices]
        else:
            labels = None
        
        images = generator(z, labels)
        all_fake.append(images.cpu())
        num_generated += current_batch
        pbar.update(current_batch)
    
    pbar.close()
    all_fake = torch.cat(all_fake, dim=0)[:num_samples]
    
    # Load real samples
    print("Loading real samples...")
    all_real = []
    for batch_data in tqdm(dataloader):
        images = batch_data[0]
        if images.max() > 1:
            images = images.float() / 127.5 - 1
        all_real.append(images)
        if len(torch.cat(all_real, dim=0)) >= num_samples:
            break
    
    all_real = torch.cat(all_real, dim=0)[:num_samples]
    
    # Compute FID
    print("\nComputing FID...")
    fid_calc = FIDCalculator(device=device)
    fid = fid_calc(all_real, all_fake, batch_size=batch_size)
    
    # Summary
    print("\n" + "=" * 70)
    print("GFGW-FM CIFAR-10 Evaluation Results")
    print("=" * 70)
    print(f"Model Parameters:  {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"NFE (Sampling):    1 (one-step generation)")
    print(f"FID (↓):           {fid:.2f}")
    print(f"Num Samples:       {num_samples}")
    print("=" * 70)
    
    # Compare with table
    print("\n📊 Comparison with Diffusion + Distillation methods:")
    print("-" * 70)
    print("Method                                    #Params  NFE  FID")
    print("-" * 70)
    print("GFGW-FM (Ours, from scratch)             61.8M    1    {:.2f}".format(fid))
    print("Consistency Distillation (EDM teacher)   55.7M    1    3.55")
    print("SlimFlow (EDM teacher)                   27.9M    1    4.53")
    print("1-Rectified Flow (+distill)              61.8M    1    6.18")
    print("2-Rectified Flow (+distill)              61.8M    1    4.85")
    print("-" * 70)
    
    return {
        'fid': fid,
        'num_params': num_params,
    }


def main():
    parser = argparse.ArgumentParser(description='GFGW-FM Simple Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to CIFAR-10 data')
    parser.add_argument('--num-samples', type=int, default=50000,
                        help='Number of samples for evaluation')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    
    args = parser.parse_args()
    
    evaluate(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()

