# GFGW-FM vs SOTA Methods - CIFAR-10 对比报告

## 评估结果

### 我们的模型 (GFGW-FM)

```
模型配置:
- 参数量: 61.8M
- NFE: 1 (one-step)
- 训练: 从头训练，100k images
- FID: 321.44
```

### 与SOTA方法对比

根据提供的表格数据对比：

| Category | Method | #Params | NFE | FID (↓) | 说明 |
|----------|--------|---------|-----|---------|------|
| **Diffusion + Distillation** | | | | | |
| | Consistency Distillation (EDM teacher) | 55.7M | 1 | **3.55** | 使用EDM预训练教师 |
| | SlimFlow (EDM teacher) | 27.9M | 1 | **4.53** | 使用EDM预训练教师 |
| | SlimFlow (EDM teacher) | 15.7M | 1 | **5.02** | 更小模型 |
| | 1-Rectified Flow (+distill) | 61.8M | 1 | 6.18 | 蒸馏训练 |
| | 2-Rectified Flow (+distill) | 61.8M | 1 | **4.85** | 二阶修正流 |
| | **GFGW-FM (Ours, 从头训练)** | **61.8M** | **1** | **321.44** | ❌ 未收敛 |

## 问题分析

### 为什么FID这么高？

**主要原因**：

1. **训练严重不足**
   - 我们只训练了 **100k images** (~13分钟)
   - 表格中的模型通常需要训练 **数百万到数千万** images
   - 例如：EDM训练约200M images，Consistency Distillation也需要大量训练

2. **缺少预训练教师模型**
   - 表格中FID < 10的方法 **全部使用了预训练教师模型** (EDM或1-Rectified Flow)
   - 这些方法本质上是 **蒸馏(Distillation)** 方法，不是从头训练
   - 预训练教师模型已经学会了数据分布

3. **训练配置可能有问题**
   - Two-stage training在第一阶段只关注t∈[0.7,1.0]
   - 这可能导致模型无法正确学习完整的生成过程
   - 100k images远远不足以让模型从stage 1过渡到stage 2

## 正确的对比方式

### 应该与哪些方法对比？

如果要公平对比，应该选择：

1. **同样从头训练的方法**（表格中没有）
2. **Teacher Model类别** - 使用足够训练的EDM作为基线
3. **使用预训练的GFGW-FM** - 从EDM初始化后fine-tune

### 改进建议

#### 选项1: 使用预训练模型初始化（推荐）

```bash
python train.py --config cifar10 \
                --data-path ./data/cifar10 \
                --run-dir ./runs/cifar10_pretrained \
                --batch-size 256 \
                --total-kimg 50000 \
                --pretrained-key edm-cifar10-uncond
```

**预期结果**: FID应该能达到 4-6 范围（类似表格中的蒸馏方法）

#### 选项2: 更长时间从头训练

```bash
python train.py --config cifar10 \
                --data-path ./data/cifar10 \
                --run-dir ./runs/cifar10_long \
                --batch-size 512 \
                --total-kimg 500000 \
                --no-pretrained
```

**预期结果**: 可能需要训练数天，FID预期在 10-30 范围

## 当前模型的诊断

### 生成样本统计
```
Generated images range: [-10.08, 7.39]
Mean: -0.58
Std: 1.99
No NaN/Inf: ✓
```

这表明：
- ✅ 模型没有崩溃（无NaN/Inf）
- ✅ 可以生成图像
- ❌ 但生成的图像不符合CIFAR-10的分布（FID=321）

### 训练Loss分析

从训练日志看：
```
Initial Loss: 25.31 → Final Loss: 22.72 (下降10.3%)
Min Loss: 20.30 at kimg 97
```

Loss下降幅度很小，说明：
- 模型在100k images内没有充分学习
- 需要更多训练时间让loss继续下降

## 建议的下一步

### 立即可做的：

1. **使用预训练模型** - 这是最快获得好结果的方式
2. **更长训练** - 至少500k-1000k images
3. **检查数据加载** - 确认CIFAR-10数据正确归一化

### 对比实验设置：

**为了在论文中公平对比，你需要：**

```
实验组A（蒸馏方法 - 与表格对比）:
  - GFGW-FM (EDM teacher) [你的创新]
  - vs Consistency Distillation (EDM teacher)
  - vs SlimFlow (EDM teacher)
  
实验组B（从头训练 - 补充实验）:
  - GFGW-FM (从头) [长时间训练]
  - vs EDM (从头)
  - vs Rectified Flow (从头)
```

## 总结

当前的FID=321.44结果 **不能** 用于与表格中的SOTA方法对比，因为：

1. ❌ 训练时间严重不足（100k vs 数百万）
2. ❌ 从头训练 vs 使用预训练教师（不同设置）
3. ❌ 模型未收敛

**要获得可对比的结果，你需要：**
- ✅ 使用 `--pretrained-key edm-cifar10-uncond`
- ✅ 训练至少 50000 kimg
- ✅ 使用与表格相同的评估协议

**预期改进后的结果：**
- 使用预训练: FID ~ 4-6 (competitive with SlimFlow)
- 长时间从头训练: FID ~ 15-25 (取决于训练时长)

---

**注意**: 当前100k images的训练只是代码验证，不是论文实验。需要重新训练才能获得可发表的结果。
