import torch
import torch.nn as nn
from monai.networks.blocks import UnetResBlock, UnetUpBlock
from utils.LPMoE.lpmoe_components import DMLP_3D, BDI_Adapter_3D


class LPMoE_VesselNet(nn.Module):
    # 🔥 将默认 filters 修改为 6 层，完美对齐 dyn_unet_base.yaml
    def __init__(self, spatial_dims=3, in_channels=1, out_channels=1, filters=(32, 64, 128, 256, 320, 320)):
        super().__init__()
        self.filters = filters

        # ==================== 编码器 (6个Block) ====================
        self.encoder_blocks = nn.ModuleList([
            UnetResBlock(spatial_dims, in_channels, filters[0], kernel_size=3, stride=1, norm_name="instance"),
            # Skip 0
            UnetResBlock(spatial_dims, filters[0], filters[1], kernel_size=3, stride=2, norm_name="instance"),  # Skip 1
            UnetResBlock(spatial_dims, filters[1], filters[2], kernel_size=3, stride=2, norm_name="instance"),  # Skip 2
            UnetResBlock(spatial_dims, filters[2], filters[3], kernel_size=3, stride=2, norm_name="instance"),  # Skip 3
            UnetResBlock(spatial_dims, filters[3], filters[4], kernel_size=3, stride=2, norm_name="instance"),  # Skip 4
            UnetResBlock(spatial_dims, filters[4], filters[5], kernel_size=3, stride=2, norm_name="instance")
            # Bottleneck (第6层)
        ])

        # ==================== 专家与适配器 (对应5层Skip) ====================
        self.dmlp_convs = nn.ModuleList([
            nn.Conv3d(in_channels, filters[0], 3, 1, 1),
            nn.Conv3d(filters[0], filters[1], 3, 2, 1),
            nn.Conv3d(filters[1], filters[2], 3, 2, 1),
            nn.Conv3d(filters[2], filters[3], 3, 2, 1),
            nn.Conv3d(filters[3], filters[4], 3, 2, 1)  # 🔥 新增第5层特征提取
        ])

        self.dmlp_experts = nn.ModuleList([
            DMLP_3D(filters[0]),
            DMLP_3D(filters[1]),
            DMLP_3D(filters[2]),
            DMLP_3D(filters[3]),
            DMLP_3D(filters[4])  # 🔥 新增第5层专家
        ])

        self.bdi_adapters = nn.ModuleList([
            BDI_Adapter_3D(filters[0]),
            BDI_Adapter_3D(filters[1]),
            BDI_Adapter_3D(filters[2]),
            BDI_Adapter_3D(filters[3]),
            BDI_Adapter_3D(filters[4])  # 🔥 新增第5层适配器
        ])

        # ==================== 解码器 (5个Block) ====================
        self.decoder_blocks = nn.ModuleList([
            # Dec 0: 接收 Bottleneck(filters[5]) 和 Skip 4(filters[4])
            UnetUpBlock(spatial_dims, filters[5], filters[4], kernel_size=3, stride=2, upsample_kernel_size=2,
                        norm_name="instance"),
            # Dec 1: 接收 Dec 0 输出(filters[4]) 和 Skip 3(filters[3])
            UnetUpBlock(spatial_dims, filters[4], filters[3], kernel_size=3, stride=2, upsample_kernel_size=2,
                        norm_name="instance"),
            # Dec 2: 接收 Dec 1 输出(filters[3]) 和 Skip 2(filters[2])
            UnetUpBlock(spatial_dims, filters[3], filters[2], kernel_size=3, stride=2, upsample_kernel_size=2,
                        norm_name="instance"),
            # Dec 3: 接收 Dec 2 输出(filters[2]) 和 Skip 1(filters[1])
            UnetUpBlock(spatial_dims, filters[2], filters[1], kernel_size=3, stride=2, upsample_kernel_size=2,
                        norm_name="instance"),
            # Dec 4: 接收 Dec 3 输出(filters[1]) 和 Skip 0(filters[0])
            UnetUpBlock(spatial_dims, filters[1], filters[0], kernel_size=3, stride=2, upsample_kernel_size=2,
                        norm_name="instance")
        ])

        self.out_conv = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        f_u_skips, f_s, f_u = [], x, x
        skel_preds = []

        # 🔥 遍历 5 层浅层到深层特征，提取 Skip Connection
        for i in range(5):
            f_u = self.encoder_blocks[i](f_u)

            # 支路进入特征提取
            f_s = self.dmlp_convs[i](f_s)
            f_s, skel = self.dmlp_experts[i](f_s)
            skel_preds.append(skel)

            f_u, f_s = self.bdi_adapters[i](f_u, f_s)
            f_u_skips.append(f_u)

        # 🔥 经过第 6 层最底层的瓶颈层 (Bottleneck)
        dec_out = self.encoder_blocks[5](f_u)

        # 🔥 5 层解码器，倒序拼接 Skip Connections
        for i in range(5):
            # f_u_skips 索引范围是 0~4，倒序读取为 4, 3, 2, 1, 0
            dec_out = self.decoder_blocks[i](dec_out, f_u_skips[4 - i])

        final_out = self.out_conv(dec_out)

        if self.training:
            return {"vol": [final_out], "skel": skel_preds}
        return final_out