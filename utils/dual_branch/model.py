import torch
import torch.nn as nn
import copy
import logging

logger = logging.getLogger(__name__)


class DualStreamDynUNet(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model

        if hasattr(base_model, 'upsamples'):
            # 结构深拷贝
            self.aux_upsamples = copy.deepcopy(base_model.upsamples)
            self.aux_output_block = copy.deepcopy(base_model.output_block)
        else:
            raise AttributeError("Base model 缺少 'upsamples' 属性。")

        # 🌟 核心：打破“同卵双胞胎”魔咒，对辅分支进行独立的 Kaiming 初始化
        self._reinit_aux_decoder()

        # 定义 3D 扰动层 (用于辅分支)
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def _reinit_aux_decoder(self):
        """对辅助解码器进行独立的 Kaiming 初始化，确保差异性起点"""
        for m in self.aux_upsamples.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        for m in self.aux_output_block.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if not self.training:
            return self.base_model(x)

        # 1. 共享编码器 (Shared Encoder)
        in_layer = self.base_model.input_block(x)
        skips = [in_layer]
        features = in_layer
        for down_layer in self.base_model.downsamples:
            features = down_layer(features)
            skips.append(features)
        bottleneck = self.base_model.bottleneck(features)

        # 2. 主分支解码 (Main Decoder) - 🌟 非对称：保持特征 100% 纯净
        x1 = bottleneck
        for i, up_layer in enumerate(self.base_model.upsamples):
            x1 = up_layer(x1, skips[-(i + 1)])
        out1 = self.base_model.output_block(x1)

        # 3. 辅分支解码 (Auxiliary Decoder) - 🌟 非对称：施加 Dropout 扰动
        x2 = self.dropout(bottleneck)
        for i, up_layer in enumerate(self.aux_upsamples):
            x2 = up_layer(x2, skips[-(i + 1)])
        out2 = self.aux_output_block(x2)

        return out1, out2