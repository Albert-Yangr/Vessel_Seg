import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricConv3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_x = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels)
        self.conv_y = nn.Conv3d(channels, channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), groups=channels)
        self.conv_z = nn.Conv3d(channels, channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv_x(x) + self.conv_y(x) + self.conv_z(x))


class PseudoSnakeConv3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.tubular_x = nn.Conv3d(channels, channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=channels)
        self.tubular_y = nn.Conv3d(channels, channels, kernel_size=(1, 5, 1), padding=(0, 2, 0), groups=channels)
        self.tubular_z = nn.Conv3d(channels, channels, kernel_size=(1, 1, 5), padding=(0, 0, 2), groups=channels)

        # 🔥 科学改进：去除全局池化，使用纯 1x1 卷积保留局部体素的空间路由信息
        self.router = nn.Sequential(
            nn.Conv3d(channels, 3 * channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, x):
        feat_x, feat_y, feat_z = self.tubular_x(x), self.tubular_y(x), self.tubular_z(x)
        # weights 形状: (B, 3*C, D, H, W)
        weights = self.router(x)
        w_x, w_y, w_z = torch.split(weights, x.shape[1], dim=1)
        out = w_x * feat_x + w_y * feat_y + w_z * feat_z
        return self.act(out)


class TASA_Block(nn.Module):
    def __init__(self, channels, reduction=4, expert_type='asymmetric'):
        super().__init__()
        mid_channels = max(channels // reduction, 16)

        self.squeeze = nn.Sequential(
            nn.Conv3d(channels, mid_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(mid_channels),
            nn.GELU()
        )

        expert_type = expert_type.lower()
        if expert_type == 'asymmetric':
            self.expert = AsymmetricConv3d(mid_channels)
        elif expert_type == 'snake':
            self.expert = PseudoSnakeConv3D(mid_channels)
        else:
            # 省略其他，你可以自己加回去
            self.expert = AsymmetricConv3d(mid_channels)

        self.expand = nn.Sequential(
            nn.Conv3d(mid_channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.GELU()
        )

        self.skel_head = nn.Conv3d(channels, 1, kernel_size=1)
        self.zero_conv = nn.Conv3d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        if self.zero_conv.bias is not None:
            nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x):
        expert_feat = self.expand(self.expert(self.squeeze(x)))

        # 1. 拓扑专家预测当前层的骨架连通性 logits
        skel_logits = self.skel_head(expert_feat)

        # 2. 转换为 0~1 的概率，作为空间注意力门控图 (Spatial Attention Gate)
        skel_prob = torch.sigmoid(skel_logits)

        # 🔥 科学改进：真正的任务解耦注意力融合
        # 只有拓扑专家认为是骨架的地方 (skel_prob 接近 1)，才允许零卷积注入特征！
        injected_feat = self.zero_conv(expert_feat) * skel_prob

        return injected_feat, skel_logits