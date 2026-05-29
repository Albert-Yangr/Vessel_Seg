import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveDynUNet(nn.Module):
    def __init__(self, base_model, layer_channels_list, proj_dim=32):
        """
        [Stage 2] 多尺度独立对比学习模型 (Multi-scale Independent Contrast)
        基于 ECCV 2022 论文: Multi-scale and Cross-scale Contrastive Learning
        """
        super().__init__()
        self.base_model = base_model
        self.features = {}
        # 抓取倒数第1(浅), 2(中), 3(深) 层
        self.hook_indices = [1, 2, 3]

        # ====================================================
        # 1. 独立投影头 (Independent Projectors)
        # 不做融合，每层一个独立的 Projection Head
        # ====================================================
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_ch, in_ch, kernel_size=1, bias=False),
                nn.InstanceNorm3d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_ch, proj_dim, kernel_size=1)
            ) for in_ch in layer_channels_list
        ])

        self._register_hooks()

    def _hook_generator(self, idx):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.features[idx] = output
            elif isinstance(output, (tuple, list)):
                self.features[idx] = output[0]

        return hook_fn

    def _register_hooks(self):
        if not hasattr(self.base_model, "upsamples"): return
        total_layers = len(self.base_model.upsamples)
        for i in self.hook_indices:
            layer_idx = total_layers - i
            if layer_idx >= 0:
                self.base_model.upsamples[layer_idx].register_forward_hook(self._hook_generator(i))

    def forward(self, x):
        self.features = {}
        logits = self.base_model(x)

        if self.training:
            # 检查是否成功抓取到特征
            if len(self.features) != len(self.hook_indices):
                return logits, None

            # ====================================================
            # 2. 返回特征列表 (List of Features)
            # ====================================================
            proj_feats_list = []

            # 遍历每一层：提取 -> 投影 -> 归一化
            for i, idx in enumerate(self.hook_indices):
                raw_feat = self.features[idx]

                # 通过对应的投影头
                proj = self.projectors[i](raw_feat)

                # 归一化 (关键步骤)
                proj = F.normalize(proj, p=2, dim=1)

                proj_feats_list.append(proj)

            # 🔥🔥🔥 关键：必须返回列表，而不是 Tensor 🔥🔥🔥
            return logits, proj_feats_list
        else:
            return logits