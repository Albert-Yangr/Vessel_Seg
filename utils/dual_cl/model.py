import torch
import torch.nn as nn
import copy

class DualStreamDynUNet(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model

        if hasattr(base_model, 'upsamples'):
            self.aux_upsamples = copy.deepcopy(base_model.upsamples)
            self.aux_output_block = copy.deepcopy(base_model.output_block)
        else:
            raise AttributeError("Base model requires 'upsamples'.")

        self._reinit_aux_decoder()
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def _reinit_aux_decoder(self):
        for m in self.aux_upsamples.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

        for m in self.aux_output_block.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        if not self.training:
            return self.base_model(x)

        # Encoder
        in_layer = self.base_model.input_block(x)
        skips = [in_layer]
        features = in_layer
        for down_layer in self.base_model.downsamples:
            features = down_layer(features)
            skips.append(features)
        bottleneck = self.base_model.bottleneck(features)

        # Main Decoder
        x1 = bottleneck
        for i, up_layer in enumerate(self.base_model.upsamples):
            x1 = up_layer(x1, skips[-(i + 1)])
        out1 = self.base_model.output_block(x1)

        # Aux Decoder
        x2 = self.dropout(bottleneck)
        for i, up_layer in enumerate(self.aux_upsamples):
            x2 = up_layer(x2, skips[-(i + 1)])
        out2 = self.aux_output_block(x2)

        # 🌟 修改：支持返回倒数第二层特征，供对比学习使用
        if return_features:
            return (out1, out2), (x1, x2)
        return out1, out2