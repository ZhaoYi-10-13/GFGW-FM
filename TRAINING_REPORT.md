# GFGW-FM CIFAR-10 训练报告

## 训练配置

### 硬件环境
- **GPU**: NVIDIA A100-PCIE-40GB
- **CUDA**: 12.8
- **PyTorch**: 2.9.1+cu128

### 训练参数
- **数据集**: CIFAR-10 (50,000张训练图像)
- **Batch Size**: 256 (实际batch_gpu=32，使用8个累积步骤)
- **图像分辨率**: 32×32
- **训练时长**: 100k images (100 kimg)
- **模型参数量**: 61,805,827
- **初始化方式**: 从头训练 (无预训练模型)

### 模型架构
- **Generator**: U-Net based OneStepGenerator
  - Model channels: 128
  - Channel multipliers: (1, 2, 2, 2)
  - Attention resolutions: (16,)
  - Dropout: 0.1
- **Feature Extractor**: DINOv2 ViT-S/14 (384维特征)
- **Memory Bank Size**: 49,984 features (2048 per OT batch)

### 优化器设置
- **Optimizer**: AdamW
- **Learning Rate**: 起始 1e-4，通过cosine schedule调整
  - 最终LR: 3.124e-05
- **Weight Decay**: 0.0
- **Betas**: (0.9, 0.999)
- **Mixed Precision**: 开启 (AMP)
- **Gradient Clipping**: 1.0

### Loss配置
- **Flow Loss**: Pseudo-Huber loss
- **Feature Loss**: Cosine similarity based
- **LPIPS Loss**: VGG-based perceptual loss (weight: 0.5)
- **Structure Loss**: 距离矩阵保持loss
- **Boundary Loss**: 边界条件enforcement

### OT (Optimal Transport) 设置
- **FGW Lambda**: 0.1 → 0.125 (annealing)
- **Epsilon**: 0.100 → 0.087 (adaptive decay)
- **Sinkhorn Iterations**: 50
- **FGW Iterations**: 10

### 训练策略
- **Two-Stage Training**: 开启
  - Stage 1: t_range (0.7, 1.0) - 前10k images
  - Stage 2: 全时间范围
- **Time Sampling**: logit_student_t distribution
- **Data Augmentation**: Horizontal flip
- **EMA**: 半衰期 500k images

## 训练结果

### Loss曲线趋势
- **初始Loss** (kimg 1): 25.31
- **最终Loss** (kimg 100): 22.72
- **Loss下降幅度**: ~10%

#### 分项Loss (kimg 100):
- Flow Loss: 22.26
- Feature Loss: 0.078

### 训练效率
- **平均速度**: 约 7.8-8.2 秒/kimg
- **总训练时间**: 约 13分钟
- **GPU利用率**: 53%
- **显存使用**: 16.6 GB / 40 GB

### 保存的文件
```
runs/cifar10_test/
├── checkpoint_latest.pt  (1.2 GB)
└── stats.jsonl          (24 KB)
```

## 发现的问题与修复

### 1. Structure Loss矩阵维度不匹配
**问题**: coupling矩阵 (batch_size, memory_size) 与 D_real矩阵 (batch_size, batch_size) 维度不匹配

**修复**: 
- 在train.py中添加`features_memory`参数传递完整的memory bank features
- 在loss函数中正确处理：从memory features中根据coupling选择matched features计算结构距离

### 2. Loss返回类型混合
**问题**: 某些loss返回float而非tensor，导致.item()调用失败

**修复**: 在train_step返回时添加类型检查
```python
return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
```

### 3. 参数传递链路不完整
**问题**: `ComprehensiveFlowLoss.forward()`没有传递`features_memory`给`main_loss`

**修复**: 在wrapper forward方法中添加features_memory参数传递

## 代码改进总结

修改的文件:
1. `/root/GFGW-FM/train.py` - 添加features_memory参数传递和类型处理
2. `/root/GFGW-FM/losses/flow_matching.py` - 更新forward签名，支持features_memory
3. `/root/GFGW-FM/losses/advanced_losses.py` - 修复structure loss计算逻辑

## 训练稳定性

- ✅ 训练全程无错误中断
- ✅ Loss稳定下降，无NaN或Inf
- ✅ GPU内存使用稳定，无OOM
- ✅ 保存checkpoint成功

## 建议

### 对于更长时间的训练:
1. 可以增加`--total-kimg`到更大的值 (例如 500 或 1000)
2. 使用`--pretrained-key edm-cifar10-uncond`加载预训练模型加速收敛
3. 调整batch size到更大 (如512) 充分利用A100-40G显存

### 对于评估:
```bash
python evaluate.py --checkpoint ./runs/cifar10_test/checkpoint_latest.pt \
                   --data-path ./data/cifar10 \
                   --num-samples 50000
```

### 对于生成样本:
```bash
python sample.py --checkpoint ./runs/cifar10_test/checkpoint_latest.pt \
                 --output-dir ./samples \
                 --num-samples 1000
```

## 总结

✅ **训练成功完成**: CIFAR-10数据集上的GFGW-FM模型训练100k images已成功完成
✅ **代码修复**: 修复了3个关键bug，保证了训练的正常运行
✅ **性能良好**: GPU利用率适中，训练速度满足要求
✅ **可扩展性**: 可以轻松扩展到更大的batch size和更长的训练时间

训练日志保存在: `/root/GFGW-FM/train_full.log`
Checkpoint保存在: `/root/GFGW-FM/runs/cifar10_test/checkpoint_latest.pt`
