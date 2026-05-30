import copy
import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


class DualStreamDynUNet(nn.Module):
    """
    双分支 DynUNet 封装。

    这个类把一个普通 DynUNet 改造成 CPS 需要的双分支模型：
    - encoder、bottleneck 共享。
    - 主分支 decoder 使用 base_model 原始 decoder。
    - 辅助分支 decoder 复制一份原始 decoder，并重新初始化。

    训练阶段：
        forward(x) -> (out1, out2)
        两个输出分别进入监督损失和伪标签损失。

    验证/测试阶段：
        forward(x) -> base_model(x)
        只返回单分支结果，方便 MONAI sliding_window_inference。
    """

    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model

        if not hasattr(base_model, "upsamples"):
            raise AttributeError("Base model must expose 'upsamples' for decoder cloning.")

        # 只复制 decoder，不复制 encoder；这样两个分支共享编码表征。
        self.aux_upsamples = copy.deepcopy(base_model.upsamples)
        self.aux_output_block = copy.deepcopy(base_model.output_block)

        # 辅助 decoder 重新初始化，避免两个分支从完全相同参数开始。
        self._reinit_aux_decoder()

        # 只扰动辅助分支 bottleneck，让两个分支预测存在差异。
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def _reinit_aux_decoder(self):
        """
        重新初始化辅助 decoder。

        如果辅助 decoder 只是原样复制主 decoder，两个分支初始输出会非常接近，
        CPS 的互监督信号容易退化。这里使用 Kaiming 初始化，让辅助分支具有
        独立起点。
        """
        for m in self.aux_upsamples.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for m in self.aux_output_block.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播。

        训练时手动展开 base_model 的 encoder / decoder，以得到双分支输出。
        验证时直接调用 base_model，避免 sliding-window 推理接收到 tuple。
        """
        if not self.training:
            return self.base_model(x)

        # 1. 共享 encoder，保存 skip features 供两个 decoder 使用。
        in_layer = self.base_model.input_block(x)
        skips = [in_layer]
        features = in_layer
        for down_layer in self.base_model.downsamples:
            features = down_layer(features)
            skips.append(features)
        bottleneck = self.base_model.bottleneck(features)

        # 2. 主分支 decoder：使用 base_model 原始 decoder。
        x1 = bottleneck
        for i, up_layer in enumerate(self.base_model.upsamples):
            x1 = up_layer(x1, skips[-(i + 1)])
        out1 = self.base_model.output_block(x1)

        # 3. 辅助分支 decoder：对 bottleneck 加 dropout 后使用复制 decoder。
        x2 = self.dropout(bottleneck)
        for i, up_layer in enumerate(self.aux_upsamples):
            x2 = up_layer(x2, skips[-(i + 1)])
        out2 = self.aux_output_block(x2)

        return out1, out2
