""" Parts of the U-Net model """

# reference https://github.com/milesial/Pytorch-UNet
import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, groups=groups, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, groups=groups, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, groups=groups, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, inplace=True)


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, groups=groups,
                      stride=1, padding=((kernel_size - 1) // 2),
                      bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.groups = groups
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, groups=groups, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.groups = groups
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, groups=groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, groups, bilinear=True):
        super().__init__()
        self.groups = groups

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, groups=groups, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, groups=groups)

        self.group_cat = GroupCat(groups)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.group_cat(x2, x1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super(OutConv, self).__init__()
        self.groups = groups
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, groups=groups, kernel_size=1),
                                  nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.conv(x)


class GroupCat(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x_1, x_2):
        batch_size = x_1.shape[0]
        h_size, w_size = x_1.shape[2], x_1.shape[3]
        assert (x_1.shape[0] == x_2.shape[0])
        assert (h_size == x_2.shape[2] and w_size == x_2.shape[3])

        x_1_features = x_1.shape[1] // self.groups
        x_2_features = x_2.shape[1] // self.groups

        r_1 = x_1.reshape(batch_size,
                          self.groups, x_1_features,
                          h_size, w_size)
        r_2 = x_2.reshape(batch_size,
                          self.groups, x_2_features,
                          h_size, w_size)

        result = torch.cat([r_1, r_2], dim=2).reshape(batch_size,
                                                      self.groups * (x_1_features + x_2_features),
                                                      h_size, w_size)
        return result
