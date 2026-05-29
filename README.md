# 三维血管分割：双分支 CPS 与动态小框对比学习

本项目主要包含两条训练程序：

```text
1. dual_train：基础双分支 CPS 半监督训练
2. dyn_box_train：双分支 CPS + 连通域自适应小框对比学习
```

研究场景是三维血管分割中的切片弱监督与少样本伪标签半监督训练。少量病例具有切片级弱标注，大量病例没有人工标签，通过双分支伪标签互监督提升三维分割性能。

## 一、基础版：dual_train

入口程序：

```text
train/dual_train.py
```

配置文件：

```text
configs/train/dual_train.yaml
```

数据配置：

```text
configs/data/CAS2023.yaml
configs/data/default_finetune.yaml
```

模型配置：

```text
configs/model/dyn_unet_base.yaml
```

主要代码结构：

```text
utils/dual_branch/
├── dataset.py
│   ├── UnlabeledWeakDataset
│   │   └── 读取无标签图像 train-all，只构造 dummy mask 以适配 MONAI transform
│   └── DualStreamDataset
│       └── 将有标签样本和无标签样本组合成一个训练 batch
│
├── model.py
│   └── DualStreamDynUNet
│       ├── 共享 DynUNet encoder
│       ├── 主 decoder 输出 pred1
│       └── 辅 decoder 输出 pred2
│
├── loss.py
│   └── DualBranchLoss
│       ├── 有标签分支：两个 decoder 都计算监督 loss
│       └── 无标签分支：两个 decoder 互相生成伪标签，计算 CPS loss
│
├── simple_loss.py
│   ├── SimpleSliceLoss
│   ├── SparseSliceLoss
│   └── SimplePseudoLoss
│
└── module.py
    └── DualBranchPLModule
        ├── training_step
        ├── validation_step
        └── configure_optimizers

utils/mutil_supervision/
└── mutil_dataset.py
    ├── MutilSupervisionDataset
    │   └── 读取有标签/切片弱标注训练数据
    └── UnionDataset
        └── 读取验证或测试数据
```

### 1. 数据流

数据目录逻辑：

```text
有标签数据：
  dataset_root/train
  包含 image 和 .slice.nii.gz 切片弱标注

无标签数据：
  dataset_root/train-all
  只读取 image，不使用真实标签

验证数据：
  dataset_root/test
  包含 image 和 .label.nii.gz 完整标签
```

每个训练 batch 包含：

```text
{
  "labeled":   有标签图像 + 切片弱标注 mask,
  "unlabeled": 无标签图像 + dummy mask
}
```

无标签流决定一个 epoch 的训练长度。有标签流在每个 step 中与无标签样本配对。

### 2. 模型结构

基础模型是一个双分支 DynUNet：

```text
共享 encoder
├── 主 decoder -> pred1
└── 辅 decoder -> pred2
```

训练阶段使用两个 decoder。  
验证和推理阶段只使用主分支 `base_model`。

### 3. 损失函数

基础训练目标：

```text
total_loss = supervised_loss + pseudo_label_loss
```

有标签监督：

```text
pred1_l 和 pred2_l 都与切片弱标注 mask_l 计算监督损失。
对于 .slice.nii.gz 标签，255 表示未知区域，不参与监督。
```

无标签伪标签监督：

```text
pred1_u 生成伪标签监督 pred2_u
pred2_u 生成伪标签监督 pred1_u
```

这就是基础 Cross Pseudo Supervision（CPS）逻辑。

## 二、动态小框版：dyn_box_train

入口程序：

```text
train/dyn_box_train.py
```

配置文件：

```text
configs/train/dyn_box_train.yaml
```

主要代码结构：

```text
utils/dual_cl/
├── model.py
│   └── DualStreamDynUNet
│       ├── 共享 encoder
│       ├── 主 decoder 输出 pred1 和 feat1
│       ├── 辅 decoder 输出 pred2 和 feat2
│       └── return_features=True 时返回 decoder 特征
│
├── loss_dyn_box.py
│   ├── ComponentAdaptivePatchContrastiveLoss
│   │   ├── 从血管 anchor 找二维连通域
│   │   ├── 根据连通域计算 bbox
│   │   ├── 对 bbox 外扩 margin
│   │   ├── 裁剪任意长宽 ROI
│   │   ├── resize 到统一 roi_size
│   │   ├── 构建血管/背景 EMA 原型
│   │   ├── 计算 macro patch-level 对比损失
│   │   └── 计算 micro ROI 内血管-近背景对比损失
│   │
│   └── DualBranchLoss
│       ├── 监督 loss
│       ├── CPS 伪标签 loss
│       └── 连通域自适应对比学习 loss
│
└── module.py
    └── DualBranchPLModule
        ├── training_step
        ├── validation_step
        └── configure_optimizers
```

`dyn_box_train` 使用的数据集读取仍来自：

```text
utils/dual_branch/dataset.py
utils/dual_branch/simple_loss.py
```

### 1. 与基础版的区别

`dual_train` 中模型只返回预测图：

```text
pred1, pred2
```

`dyn_box_train` 中模型额外返回 decoder 特征：

```text
pred1, pred2
feat1, feat2
```

这些特征用于对比学习。

### 2. 动态小框对比学习思想

固定大小 16x16 小框对冠状动脉较适合，因为冠脉在正交切片上经常呈现点状或小团块。

但对于脑动脉和肺部血管，血管在切片上可能呈现：

```text
脑动脉：长条状结构
肺部血管：大块连通团块
```

固定小框可能无法完整覆盖血管结构，因此动态小框版本使用连通域形态来决定 ROI。

流程如下：

```text
1. 在切片弱标签或伪标签中采样血管 anchor 点。
2. 在当前二维切片上找到 anchor 所属连通域。
3. 计算该连通域的最小外接 bbox。
4. 按 margin_ratio 向外扩展 bbox，引入血管周围难分背景。
5. 裁剪任意长宽的 ROI。
6. 将 ROI resize 到统一大小 roi_size x roi_size。
7. 在 ROI 内提取血管特征和背景特征。
8. 计算对比学习损失。
```

### 3. 对比学习损失

动态小框模块维护两个 EMA 原型：

```text
vessel_proto：血管原型
bg_proto：背景原型
```

有标签数据：

```text
使用真实切片弱标注采样可靠血管和背景区域。
用这些可靠特征更新 vessel_proto 和 bg_proto。
```

无标签数据：

```text
warmup_epochs 之后使用伪标签区域参与对比学习。
无标签数据只计算对比损失，不更新原型。
```

对比学习包括两个层次：

```text
1. Macro 对比：
   血管 ROI 特征靠近 vessel_proto，远离 bg_proto。
   背景 ROI 特征靠近 bg_proto，远离 vessel_proto。

2. Micro 对比：
   在一个血管 ROI 内，
   血管区域特征靠近 vessel_proto，
   并远离 ROI 内的近邻背景 hard negative。
```

完整训练目标：

```text
total_loss = supervised_loss + pseudo_label_loss + contrastive_weight * contrastive_loss
```

## 三、两个训练程序的核心差异

```text
dual_train:
  基础双分支 CPS。
  只使用监督损失和伪标签损失。
  不使用 decoder 特征。
  不包含对比学习。

dyn_box_train:
  双分支 CPS + 连通域自适应小框对比学习。
  使用 decoder 特征。
  通过动态 ROI 加强血管/背景局部特征区分。
```

文件对应关系：

```text
基础 CPS:
  train/dual_train.py
  configs/train/dual_train.yaml
  utils/dual_branch/model.py
  utils/dual_branch/loss.py
  utils/dual_branch/module.py
  utils/dual_branch/dataset.py
  utils/dual_branch/simple_loss.py

动态小框 CPS:
  train/dyn_box_train.py
  configs/train/dyn_box_train.yaml
  utils/dual_cl/model.py
  utils/dual_cl/loss_dyn_box.py
  utils/dual_cl/module.py
  utils/dual_branch/dataset.py
  utils/dual_branch/simple_loss.py
```

## 四、运行方式

基础双分支 CPS：

```bash
python train/dual_train.py
```

动态小框对比学习：

```bash
python train/dyn_box_train.py
```

也可以覆盖数据集配置：

```bash
python train/dyn_box_train.py data=CAS2023 data_name=CAS2023
python train/dyn_box_train.py data=Parse data_name=Parse
python train/dyn_box_train.py data=imageCAS data_name=imageCAS
```

## 五、需要注意的实现细节

1. 基础 CPS 伪标签当前采用 0.5 阈值生成硬伪标签。

2. 动态小框版本在 `warmup_epochs` 之后才让无标签伪标签参与对比学习。

3. 动态小框配置中包含 `pseudo_confidence` 和 `use_reliable_agreement`，但当前实现主要通过两个 decoder 预测差异过滤无标签对比区域。

4. 验证阶段使用 sliding-window inference，并只评估主分支。

5. `255` 表示未知或忽略区域，不参与切片弱监督和对比学习中的有效像素计算。
