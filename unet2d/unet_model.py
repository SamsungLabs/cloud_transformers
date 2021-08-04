# reference https://github.com/milesial/Pytorch-UNet

from unet2d.unet_parts import *

import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_out, groups, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_out = n_out
        self.groups = groups
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels * self.groups, 16 * self.groups, self.groups)
        self.down1 = Down(16 * self.groups, 32 * self.groups, self.groups)
        self.down2 = Down(32 * self.groups, 64 * self.groups, self.groups)
        self.down3 = Down(64 * self.groups, 64 * self.groups, self.groups)
        self.down4 = Down(64 * self.groups, 64 * self.groups, self.groups)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, 1024)
        self.up1 = Up((64 + 64) * self.groups, 64 * self.groups, self.groups, bilinear)
        self.up2 = Up(128 * self.groups, 64 * self.groups, self.groups, bilinear)
        self.up3 = Up((64 + 32) * self.groups, 32 * self.groups, self.groups, bilinear)
        self.up4 = Up((32 + 16) * self.groups, 16 * self.groups, self.groups, bilinear)
        self.outc = OutConv(16 * self.groups, n_out * self.groups, self.groups)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = F.leaky_relu(x5 + self.linear(self.avgpool(x5).reshape(-1, 1024)).reshape(-1, 1024, 1, 1), inplace=True)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
