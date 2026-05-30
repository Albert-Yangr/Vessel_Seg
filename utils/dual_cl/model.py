import copy

import torch.nn as nn


class DualStreamDynUNet(nn.Module):
    """
    对比学习版本的双分支 DynUNet。

    结构设计：
      - encoder 和 bottleneck 共享，用同一套编码特征表示输入图像；
      - 主分支 decoder 使用 base_model 原始 decoder；
      - 辅助分支 decoder 深拷贝一份，并重新初始化；
      - 辅助分支在 bottleneck 后加入 Dropout，使两个分支产生一定预测差异。

    训练阶段：
      forward(x, return_features=False) -> (out1, out2)
      forward(x, return_features=True)  -> ((out1, out2), (feat1, feat2))

    验证/测试阶段：
      forward(x) -> base_model(x)
      只返回单分支预测，方便 sliding-window inference。
    """

    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model

        if not hasattr(base_model, "upsamples"):
            raise AttributeError("Base model requires 'upsamples'.")

        # 只复制 decoder，不复制 encoder；两个分支共享 encoder 表征。
        self.aux_upsamples = copy.deepcopy(base_model.upsamples)
        self.aux_output_block = copy.deepcopy(base_model.output_block)

        self._reinit_aux_decoder()
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def _reinit_aux_decoder(self):
        """重新初始化辅助 decoder，避免两个分支从完全相同的 decoder 参数开始训练。"""
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

    def forward(self, x, return_features=False):
        """
        前向传播。

        注意：
          - self.training=False 时直接调用 base_model，用于验证/测试；
          - self.training=True 时手动展开 DynUNet 的 encoder/bottleneck/decoder，
            这样才能让两个 decoder 分支共享 encoder，并拿到 decoder feature。
        """
        if not self.training:
            return self.base_model(x)

        # 1. 共享 encoder，并保存 skip features，供两个 decoder 分支使用。
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

        # 3. 辅助分支 decoder：对 bottleneck 加 Dropout，再走复制出来的 decoder。
        x2 = self.dropout(bottleneck)
        for i, up_layer in enumerate(self.aux_upsamples):
            x2 = up_layer(x2, skips[-(i + 1)])
        out2 = self.aux_output_block(x2)

        # x1/x2 是输出头前的 decoder feature，用于后续小框对比学习。
        if return_features:
            return (out1, out2), (x1, x2)
        return out1, out2
