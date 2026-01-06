#!/bin/bash
# ============================================================================
# GFGW-FM Training with Pretrained Initialization (ECM/TCM Best Practice)
# ============================================================================
#
# This script trains GFGW-FM on CIFAR-10 using pretrained EDM weights.
# Following the same strategy as ECM (Consistency Models Made Easy) and
# TCM (Truncated Consistency Models).
#
# Key benefits:
#   - Much faster convergence (hours instead of days)
#   - More stable training
#   - Fair comparison with other methods
#   - Same as top ECCV/ICLR papers
#
# ============================================================================

# Configuration
DATA_PATH="../datasets/cifar10-32x32.zip"  # Path to your CIFAR-10 dataset
RUN_DIR="./runs/gfgw_cifar10_pretrained"
BATCH_SIZE=128
TOTAL_KIMG=50000  # 50M images (like ECM setting)

# ============================================================================
# Option 1: Use predefined pretrained model key
# ============================================================================
python train.py \
    --config cifar10 \
    --data-path $DATA_PATH \
    --run-dir $RUN_DIR \
    --pretrained-key edm-cifar10-uncond \
    --batch-size $BATCH_SIZE \
    --total-kimg $TOTAL_KIMG

# ============================================================================
# Option 2: Use direct URL (same as above, just explicit)
# ============================================================================
# python train.py \
#     --config cifar10 \
#     --data-path $DATA_PATH \
#     --run-dir $RUN_DIR \
#     --pretrained "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
#     --batch-size $BATCH_SIZE \
#     --total-kimg $TOTAL_KIMG

# ============================================================================
# Option 3: Use local pretrained checkpoint
# ============================================================================
# python train.py \
#     --config cifar10 \
#     --data-path $DATA_PATH \
#     --run-dir $RUN_DIR \
#     --pretrained "./pretrained/edm-cifar10-32x32-uncond-vp.pkl" \
#     --batch-size $BATCH_SIZE \
#     --total-kimg $TOTAL_KIMG

# ============================================================================
# Option 4: Fast training (1 GPU hour like ECM)
# Use smaller total_kimg for quick experiments
# ============================================================================
# python train.py \
#     --config cifar10 \
#     --data-path $DATA_PATH \
#     --run-dir "./runs/gfgw_cifar10_1hour" \
#     --pretrained-key edm-cifar10-uncond \
#     --batch-size 128 \
#     --total-kimg 10000  # ~10k images, should finish in ~1 hour on A100

# ============================================================================
# Option 5: From scratch (NOT recommended, for ablation only)
# ============================================================================
# python train.py \
#     --config cifar10 \
#     --data-path $DATA_PATH \
#     --run-dir "./runs/gfgw_cifar10_scratch" \
#     --no-pretrained \
#     --batch-size $BATCH_SIZE \
#     --total-kimg 200000  # Need much longer training

