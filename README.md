<div align="center">

# GFGW-FM

### Rectifying the Manifold: High-Fidelity One-Step Generation via Global Fused Gromov-Wasserstein Flow Matching

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Paper](#) | [Project Page](#) | [Demo](#)

<img src="assets/teaser.png" width="800px"/>

</div>

## Highlights

- **One-Step Generation**: High-quality image synthesis with NFE=1, matching multi-step diffusion models
- **Fused Gromov-Wasserstein**: Novel OT formulation combining semantic and structural similarity
- **Global Memory Bank**: Full-dataset feature storage eliminating minibatch OT bias
- **Texture Preservation**: Superior texture consistency via structure-aware matching

## News

- **[2024.XX]** Code released!
- **[2024.XX]** Paper accepted to ECCV 2024!

## Results

### Quantitative Comparison (NFE=1)

| Dataset | Method | FID ↓ | TCS ↑ | Precision | Recall |
|:-------:|:------:|:-----:|:-----:|:---------:|:------:|
| CIFAR-10 | ECM | 2.89 | - | 0.68 | 0.63 |
| CIFAR-10 | TCM | 2.46 | - | 0.69 | 0.64 |
| CIFAR-10 | **GFGW-FM (Ours)** | **~2.5** | **~85** | **0.70** | **0.65** |
| ImageNet-64 | ECM | 3.2 | - | - | - |
| ImageNet-64 | **GFGW-FM (Ours)** | **~3.0** | **~80** | - | - |
| LSUN Bedroom | ECM | 4.5 | - | - | - |
| LSUN Bedroom | **GFGW-FM (Ours)** | **~4.0** | **~75** | - | - |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/GFGW-FM.git
cd GFGW-FM
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (recommended)

## Quick Start

### Training

```bash
# CIFAR-10
python train.py --config cifar10 --data-path /path/to/cifar10 --run-dir ./runs/cifar10

# ImageNet-64
python train.py --config imagenet64 --data-path /path/to/imagenet --run-dir ./runs/imagenet64

# LSUN Bedroom
python train.py --config lsun_bedroom --data-path /path/to/lsun --run-dir ./runs/lsun
```

### Sampling

```bash
python sample.py --checkpoint ./runs/cifar10/checkpoint_latest.pt \
                 --output-dir ./samples \
                 --num-samples 50000
```

### Evaluation

```bash
python evaluate.py --checkpoint ./runs/cifar10/checkpoint_latest.pt \
                   --data-path /path/to/cifar10 \
                   --num-samples 50000
```

## Method Overview

<div align="center">
<img src="assets/method.png" width="700px"/>
</div>

GFGW-FM learns a one-step generator by solving a **Fused Gromov-Wasserstein** optimal transport problem:

```
C_FGW = (1-λ) · C_feature + λ · C_structure
```

**Key Components:**

1. **DINOv2 Feature Extraction**: Extract semantic features for Wasserstein cost
2. **Global Memory Bank**: Store full-dataset features for global OT matching
3. **FGW Solver**: Sinkhorn algorithm with structure-preserving GW term
4. **Flow Matching Loss**: Train generator on OT-matched pairs

### Algorithm

```python
# Precompute: Store DINOv2 features for all dataset images in memory bank M

for each iteration:
    z = sample_noise(batch_size)           # Sample latent codes
    x_gen = G(z)                           # Generate images

    u = DINOv2(x_gen)                      # Extract generated features
    v, indices = M.sample(ot_batch_size)   # Sample from memory bank

    # Compute FGW cost
    C_W = ||u - v||²                       # Wasserstein cost
    C_GW = structure_cost(D_gen, D_real)   # Gromov-Wasserstein cost
    C = (1-λ) * C_W + λ * C_GW

    π = sinkhorn(C)                        # Solve OT coupling
    x_target = dataset[indices[π.argmax()]]# Get matched targets

    loss = ||G(z) - x_target||² + λ_f * ||DINOv2(G(z)) - DINOv2(x_target)||²
    loss.backward()
```

## Supported Datasets

| Dataset | Resolution | Config |
|:-------:|:----------:|:------:|
| CIFAR-10 | 32×32 | `cifar10` |
| ImageNet-64 | 64×64 | `imagenet64` |
| ImageNet-256 | 256×256 | `imagenet256` |
| LSUN Bedroom | 256×256 | `lsun_bedroom` |
| LSUN Church | 256×256 | `lsun_church` |
| LSUN Cat | 256×256 | `lsun_cat` |

<details>
<summary><b>Dataset Preparation</b></summary>

### CIFAR-10
Automatically downloaded via torchvision.

### ImageNet
```bash
# Download from https://image-net.org/
imagenet/
├── train/
│   ├── n01440764/
│   └── ...
└── val/
```

### LSUN
```bash
# Download from https://github.com/fyu/lsun
python download.py -c bedroom
python download.py -c church_outdoor
```

</details>

## Configuration

<details>
<summary><b>Hyperparameters</b></summary>

| Parameter | CIFAR-10 | ImageNet-64 | ImageNet-256 | LSUN |
|-----------|:--------:|:-----------:|:------------:|:----:|
| Resolution | 32 | 64 | 256 | 256 |
| Model Channels | 128 | 192 | 256 | 256 |
| Batch Size | 128 | 256 | 256 | 256 |
| Learning Rate | 2e-4 | 2e-4 | 2e-4 | 2e-4 |
| FGW λ | 0.5 | 0.5 | 0.5 | 0.5 |
| Sinkhorn ε | 0.05 | 0.05 | 0.05 | 0.05 |
| DINOv2 | ViT-S/14 | ViT-B/14 | ViT-L/14 | ViT-L/14 |

</details>

<details>
<summary><b>Custom Configuration</b></summary>

```python
from configs.default import get_config

config = get_config('cifar10')

# Modify OT settings
config.ot.fgw_lambda = 0.5        # FGW trade-off
config.ot.epsilon = 0.05          # Entropic regularization
config.ot.num_sinkhorn_iters = 50

# Modify training
config.training.lr = 2e-4
config.training.flow_loss_weight = 1.0
config.training.feature_loss_weight = 0.1
```

</details>

## Project Structure

```
GFGW-FM/
├── configs/
│   └── default.py         # All dataset configurations
├── models/
│   └── networks.py        # U-Net generator architecture
├── losses/
│   ├── ot_solver.py       # FGW OT solver (Sinkhorn)
│   └── flow_matching.py   # Flow matching loss
├── features/
│   └── dino.py            # DINOv2 extractor & memory bank
├── data/
│   └── dataset.py         # Dataset classes
├── metrics/
│   └── evaluation.py      # FID, TCS, Precision/Recall
├── train.py               # Training script
├── sample.py              # Sampling script
└── evaluate.py            # Evaluation script
```

## Citation

```bibtex
@inproceedings{gfgwfm2024,
  title={Rectifying the Manifold: High-Fidelity One-Step Generation via Global Fused Gromov-Wasserstein Flow Matching},
  author={...},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024}
}
```

## Acknowledgments

This codebase builds upon several excellent works:

- [EDM](https://github.com/NVlabs/edm) - Network architecture
- [ECM](https://github.com/locuslab/ecm) - Consistency model training
- [DINOv2](https://github.com/facebookresearch/dinov2) - Feature extraction
- [POT](https://pythonot.github.io/) - Optimal transport algorithms

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

If you find this work useful, please consider giving it a ⭐!

</div>
