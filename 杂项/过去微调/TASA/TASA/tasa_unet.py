import torch
import torch.nn as nn
from monai.networks.nets import DynUNet
from utils.TASA.tasa_components import TASA_Block


class TASA_VesselNet(nn.Module):
    def __init__(self, base_model: nn.Module, expert_type="snake"):
        super().__init__()
        # 1. 吞噬基础大模型 (原封不动地保留其结构)
        self.base_model = base_model

        # 2. 匹配 VesselFM 真实的 YAML 通道配置 (6 层)
        self.filters = [32, 64, 128, 256, 320, 320]
        num_levels = len(self.filters)

        if isinstance(expert_type, str):
            expert_types = [expert_type] * num_levels
        elif isinstance(expert_type, (list, tuple)) or type(expert_type).__name__ == 'ListConfig':
            expert_types = list(expert_type)

        # 3. 动态初始化 TASA 拓扑专家旁支
        self.adapters = nn.ModuleList([
            TASA_Block(f, expert_type=t) for f, t in zip(self.filters, expert_types)
        ])

        # 4. 核心容器：用于在 forward 过程中收集各层生成的骨架预测
        self.skel_preds = []
        self.hooks = []

        # 5. 启动寄生劫持！
        self._register_hooks()

    def _register_hooks(self):
        """
        核心魔法：顺着 MONAI 的链表结构爬行，给每个下采样层打上钩子。
        """
        down_blocks = []
        curr = self.base_model.skip_layers

        # 遍历嵌套的 DynUNetSkipLayer (俄罗斯套娃)
        # 🔥 修复点：MONAI 内部的属性名叫 downsample，而不是 downsampling
        while hasattr(curr, 'next_layer'):
            down_blocks.append(curr.downsample)
            curr = curr.next_layer

        # 加上最底层的 bottleneck (它没有 next_layer)
        down_blocks.append(curr)

        # 为每一个 down_block 注册一个前向传播钩子
        for i, block in enumerate(down_blocks):
            def make_hook(idx):
                def hook(module, inputs, output):
                    # output 是基础大模型 downsample 出来的原始特征
                    # 把它送进对应的 TASA 专家进行处理
                    injected_feat, skel_pred = self.adapters[idx](output)

                    # 收集骨架预测，用于后面计算 clDice Loss
                    self.skel_preds.append(skel_pred)

                    # ✨ 关键：Hook 可以直接修改特征流！
                    # 加上 TASA 的注入特征后，全新的特征会继续往大模型深处流淌
                    return output + injected_feat

                return hook

            # 挂载钩子
            h = block.register_forward_hook(make_hook(i))
            self.hooks.append(h)

    def freeze_backbone(self):
        """
        定向冻结策略：
        1. 解冻整个网络（释放 Decoder 和分类头的领域适应能力）
        2. 精准遍历套娃结构，仅将 Encoder（下采样部分）重新上锁。
        """
        # 1. 默认全部解冻（此时整个大模型 + TASA 都能训练）
        for param in self.base_model.parameters():
            param.requires_grad = True

        # 2. 顺着链表，精准找到所有的 Encoder 层并上锁
        curr = self.base_model.skip_layers
        while hasattr(curr, 'next_layer'):
            # 仅冻结下采样模块 (Encoder)
            for param in curr.downsample.parameters():
                param.requires_grad = False
            curr = curr.next_layer

        # 3. 冻结最底层的 Bottleneck
        for param in curr.parameters():
            param.requires_grad = False

        # 此时，网络的状态是：
        # Encoder (冻结) -> 保护血管通用特征
        # Decoder (训练) -> 适应 CAS2023 局部域
        # TASA (训练) -> 修复细微拓扑断裂

    def forward(self, x):
        # 每次前向传播前，清空收集器
        self.skel_preds = []

        # 只需要让基础模型跑一遍前向传播，Hook 会自动在后台运作触发 TASA！
        out = self.base_model(x)

        if self.training:
            # 🔥 兼容升级：处理 MONAI 可能因为深监督返回 list/tuple 的情况
            if isinstance(out, (tuple, list)):
                return {"vol": list(out), "skel": self.skel_preds}
            return {"vol": [out], "skel": self.skel_preds}

        # 验证/测试时直接返回主预测掩码
        if isinstance(out, (tuple, list)):
            return out[0]
        return out