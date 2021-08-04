# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch

import torch.nn as nn
import torch.nn.functional as F


class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, groups=groups,
                      stride=1, padding=((kernel_size - 1) // 2),
                      bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, groups=groups, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, groups=groups, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, groups=groups, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, inplace=True)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, groups=1):
        super(Upsample3DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size,
                               groups=groups, stride=stride, padding=0,
                               output_padding=0, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecorder(nn.Module):
    def __init__(self, groups):
        super(EncoderDecorder, self).__init__()

        self.groups = groups

        self.encoder_pool0 = Pool3DBlock(2)
        self.encoder_res0 = Res3DBlock(32 * groups, 32 * groups, groups=groups)
        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32 * groups, 64 * groups, groups=groups)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64 * groups, 128 * groups, groups=groups)
        self.encoder_pool3 = Pool3DBlock(2)
        self.encoder_res3 = Res3DBlock(128 * groups, 128 * groups, groups=groups)

        self.mid_res = Res3DBlock(128 * groups, 128 * groups, groups=groups)

        self.decoder_res3 = Res3DBlock(128 * groups, 128 * groups, groups=groups)
        self.decoder_upsample3 = Upsample3DBlock(128 * groups, 128 * groups, 2, 2, groups=groups)
        self.decoder_res2 = Res3DBlock(128 * groups, 128 * groups, groups=groups)
        self.decoder_upsample2 = Upsample3DBlock(128 * groups, 64 * groups, 2, 2, groups=groups)
        self.decoder_res1 = Res3DBlock(64 * groups, 64 * groups, groups=groups)
        self.decoder_upsample1 = Upsample3DBlock(64 * groups, 32 * groups, 2, 2, groups=groups)
        self.decoder_res0 = Res3DBlock(32 * groups, 32 * groups)
        self.decoder_upsample0 = Upsample3DBlock(32 * groups, 32 * groups, 2, 2, groups=groups)

        self.skip_res0 = Res3DBlock(32 * groups, 32 * groups, groups=groups)
        self.skip_res1 = Res3DBlock(32 * groups, 32 * groups, groups=groups)
        self.skip_res2 = Res3DBlock(64 * groups, 64 * groups, groups=groups)
        self.skip_res3 = Res3DBlock(128 * groups, 128 * groups, groups=groups)

    def forward(self, x):
        skip_x0 = self.skip_res0(x)
        x = self.encoder_pool0(x)
        x = self.encoder_res0(x)

        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)

        x = self.mid_res(x)

        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        x = x + skip_x3
        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2
        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1
        x = self.decoder_res0(x)
        x = self.decoder_upsample0(x)
        x = x + skip_x0

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels, groups=1):
        super(V2VModel, self).__init__()
        self.groups = groups

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels * groups, 32 * groups, kernel_size=3, groups=self.groups),
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
        )

        self.encoder_decoder = EncoderDecorder(groups=self.groups)

        self.back_layers = nn.Sequential(
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
            Res3DBlock(32 * groups, 32 * groups, groups=self.groups),
        )

        self.output_layer = nn.Sequential(nn.Conv3d(32 * groups, output_channels * groups,
                                                    groups=groups,
                                                    kernel_size=1, stride=1, padding=0, bias=True))

        self._initialize_weights()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)

        x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        pass