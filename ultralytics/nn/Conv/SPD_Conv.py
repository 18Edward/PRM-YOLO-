######################  SPD-Conv  ####     start ###############################

import torch
import torch.nn as nn


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

######################  SPD-Conv  ####     start ###############################
# ==================== SPD-Conv 完整模块 ====================
class SPDConv(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=3, s=1):
        super().__init__()
        c2 = c2 or c1  # 如果未指定 c2，默认输出通道 = 输入通道
        self.s2d = space_to_depth(dimension=1)
        # 使用 stride=1 的卷积降通道 + 融合特征
        self.conv = nn.Sequential(
            nn.Conv2d(c1 * 4, c2, kernel_size=kernel_size, stride=s, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()  # 或 nn.ReLU()
        )

    def forward(self, x):
        x = self.s2d(x)      # [B, C, H, W] -> [B, 4C, H/2, W/2]
        x = self.conv(x)     # [B, 4C, H/2, W/2] -> [B, c2, H/2, W/2]
        return x