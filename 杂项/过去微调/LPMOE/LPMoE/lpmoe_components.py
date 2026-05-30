import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_x = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=in_channels)
        self.conv_y = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), groups=in_channels)
        self.conv_z = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), groups=in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv_x(x) + self.conv_y(x) + self.conv_z(x))


class DMLP_3D(nn.Module):
    """3D 稳定混合局部先验提取器 (精简版)"""

    def __init__(self, channels):
        super().__init__()
        # 🗑️ 移除了冗余的 expert_1 (普通 3x3)

        # Expert 1: 空洞卷积，捕捉更宽的局部上下文先验
        self.expert_1 = nn.Sequential(nn.Conv3d(channels, channels, 3, padding=2, dilation=2, groups=channels),
                                      nn.GELU())
        # Expert 2: 非对称卷积，专门针对管状结构的各向异性先验
        self.expert_2 = AsymmetricConv3d(channels, channels)

        # 稳定的全局权重打分 (现在只需评估 2 个专家)
        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, 2, kernel_size=1),  # 🔥 修改为 2 通道
            nn.Softmax(dim=1)
        )

        # 🌟 轻量级局部空间细化器
        self.spatial_refine = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

        # 🔥 核心修改：新增骨架预测头 (为了让专家支路支持拓扑监督)
        self.skel_head = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, x):
        e1, e2 = self.expert_1(x), self.expert_2(x)
        w = self.gating(x)  # [B, 2, 1, 1, 1]
        mixed_priors = w[:, 0:1] * e1 + w[:, 1:2] * e2

        # 局部掩码提亮
        spatial_mask = self.spatial_refine(mixed_priors)
        out_feat = self.proj(x + mixed_priors * spatial_mask)

        # 🔥 核心修改：同时输出特征和当前专家提取的拓扑蓝图
        skel_pred = self.skel_head(out_feat)
        return out_feat, skel_pred

class CDA_3D(nn.Module):
    """🌟 重新启用：高度稳定且低显存的 3D 通道交叉注意力"""

    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels, channels, 1)
        self.v_proj = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, primary_feat, aux_feat):
        B, C, D, H, W = primary_feat.shape
        norm_primary = self.norm(primary_feat.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, D, H, W)
        norm_aux = self.norm(aux_feat.flatten(2).transpose(1, 2)).transpose(1, 2).view(B, C, D, H, W)

        q = self.q_proj(norm_primary).flatten(2)
        k = self.k_proj(norm_aux).flatten(2)
        v = self.v_proj(norm_aux).flatten(2)

        q_norm = F.normalize(q, p=2, dim=2)
        k_norm = F.normalize(k, p=2, dim=2)

        attn = torch.bmm(q_norm, k_norm.transpose(1, 2))
        attn = F.softmax(attn * 10.0, dim=-1)

        out = torch.bmm(attn, v)
        out = out.view(B, C, D, H, W)

        return primary_feat + self.gamma * self.out_proj(out)


class CASE_3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.channel_att(x)


class BDI_Adapter_3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cda_u2s = CDA_3D(channels)
        self.cda_s2u = CDA_3D(channels)
        self.case = CASE_3D(channels)

    def forward(self, f_u, f_s):
        hat_f_u = self.cda_u2s(primary_feat=f_u, aux_feat=f_s)
        hat_f_s = self.cda_s2u(primary_feat=f_s, aux_feat=hat_f_u)
        hat_f_s = self.case(hat_f_s)
        return hat_f_u, hat_f_s