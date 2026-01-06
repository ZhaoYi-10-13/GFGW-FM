#!/bin/bash
# ============================================================================
# GFGW-FM Training on ImageNet-64 with Pretrained EDM2 Initialization
# ============================================================================
#
# This script trains GFGW-FM on ImageNet-64 using pretrained EDM2 weights.
# Following the same strategy as TCM (Truncated Consistency Models) which
# achieved 2.88 FID on ImageNet-64 with 1-step generation.
#
# Pretrained models available:
#   - edm2-img64-s: EDM2 Small (192 channels, ~280M params)
#   - edm2-img64-m: EDM2 Medium (256 channels)
#   - edm2-img64-l: EDM2 Large (320 channels)
#   - edm2-img64-xl: EDM2 XL (384 channels, used by TCM for best results)
#
# ============================================================================

# Configuration
DATA_PATH="../datasets/imagenet64x64.zip"  # Path to your ImageNet-64 dataset
RUN_DIR="./runs/gfgw_imagenet64_pretrained"
BATCH_SIZE=256
TOTAL_KIMG=200000

# ============================================================================
# Train with EDM2-S pretrained model (smaller, faster)
# ============================================================================
python train.py \
    --config imagenet64 \
    --data-path $DATA_PATH \
    --run-dir $RUN_DIR \
    --pretrained-key edm2-img64-s \
    --batch-size $BATCH_SIZE \
    --total-kimg $TOTAL_KIMG

# ============================================================================
# Train with EDM2-XL for best results (like TCM)
# Requires more GPU memory
# ============================================================================
# python train.py \
#     --config imagenet64 \
#     --data-path $DATA_PATH \
#     --run-dir "./runs/gfgw_imagenet64_xl" \
#     --pretrained-key edm2-img64-xl \
#     --batch-size 128 \
#     --total-kimg $TOTAL_KIMG

# ============================================================================
# Multi-GPU Training (recommended for ImageNet)
# ============================================================================
# torchrun --nnodes=1 --nproc_per_node=4 train.py \
#     --config imagenet64 \
#     --data-path $DATA_PATH \
#     --run-dir $RUN_DIR \
#     --pretrained-key edm2-img64-s \
#     --batch-size $BATCH_SIZE \
#     --total-kimg $TOTAL_KIMG

