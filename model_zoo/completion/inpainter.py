from torch import nn
import torch

from layers.multihead_ct import MultiHeadUnion
from layers.multihead_ct_pool import MultiHeadPool
from layers.multihead_ct_adain import MultiHeadUnionAdaIn, forward_style

from layers.v2v_groups import Res3DBlock, Pool3DBlock, Basic3DBlock
from unet2d.unet_parts import Res2DBlock


from layers.utils import AdaIn1dUpd


def sum_hack(lists):
    start = lists[0]
    for i in range(1, len(lists)):
        start += lists[i]

    return start


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_dim = 512

        self.first_process = nn.Sequential(nn.Conv1d(3, self.model_dim, kernel_size=1, bias=False),
                                           nn.BatchNorm1d(self.model_dim),
                                           nn.ReLU(inplace=True))

        self.attentions_encoder = nn.ModuleList(sum_hack([[MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4, 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[128, 32],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3]),
                                                 MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[64, 16],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3]),
                                                 MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 8],
                                                                heads=[16, 16],
                                                                tensor_sizes=[16, 8],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3])] for _ in range(4)]))

        self.pool3d = MultiHeadPool(model_dim=self.model_dim,
                                    in_feature_dim=4 * 8,
                                    heads=16,
                                    tensor_size=8,
                                    tensor_dim=3)

        self.heads_3d = 16
        self.heads_2d = 16

        self.after_pool3d = nn.Sequential(Res3DBlock(32 * self.heads_3d, 64 * self.heads_3d, groups=16),
                                          Pool3DBlock(2),
                                          Res3DBlock(64 * self.heads_3d, 64 * self.heads_3d, groups=16),
                                          Pool3DBlock(2),
                                          Res3DBlock(64 * self.heads_3d, 64 * self.heads_3d, groups=16),
                                          nn.AdaptiveAvgPool3d((1, 1, 1)))

        self.pool2d = MultiHeadPool(model_dim=self.model_dim,
                                    in_feature_dim=4 * 4,
                                    heads=16,
                                    tensor_size=16,
                                    tensor_dim=2)

        self.after_pool2d = nn.Sequential(Res2DBlock(16 * self.heads_2d, 32 * self.heads_2d, groups=16),
                                          nn.MaxPool2d(2),
                                          Res2DBlock(32 * self.heads_2d, 64 * self.heads_2d, groups=16),
                                          nn.MaxPool2d(2),
                                          Res2DBlock(64 * self.heads_2d, 64 * self.heads_2d, groups=16),
                                          nn.AdaptiveAvgPool2d((1, 1)))

        self.class_head = nn.Sequential(nn.Linear(1024 + 1024, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(inplace=True))

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, input, return_lattice=False):
        input = input.squeeze(2)

        x = self.first_process(input)

        lattices_sizes = []
        for i in range(len(self.attentions_encoder)):
            x, lattice_size = self.attentions_encoder[i](x, input)

            lattices_sizes += lattice_size

        res = x

        assert(res.size(-1) == input.size(-1))

        to_3d, lattice_size_2d = self.pool3d(res, input)
        to_2d, lattice_size_3d = self.pool2d(res, input)

        lattices_sizes += [lattice_size_2d]
        lattices_sizes += [lattice_size_3d]

        pooled_2d = self.after_pool2d(to_2d).reshape(-1, 1024)
        pooled_3d = self.after_pool3d(to_3d).reshape(-1, 1024)

        class_pred = self.class_head(torch.cat([pooled_2d, pooled_3d], dim=-1))

        return class_pred, lattices_sizes


class Model(nn.Module):
    def __init__(self, num_latent=512):
        super().__init__()

        self.model_dim = 512

        self.encoder = Encoder()

        self.mapping = nn.Sequential(nn.Linear(1024, num_latent),
                                     nn.ReLU(inplace=True))

        self.start = nn.Sequential(nn.Conv1d(in_channels=4,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(True))

        self.attentions_decoder = nn.ModuleList(sum_hack([[MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=[4, 4],
                                                                               heads=[16, 16],
                                                                               tensor_sizes=[128, 32],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3]),
                                                           MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=[4 * 4, 4 * 4],
                                                                               heads=[16, 16],
                                                                               tensor_sizes=[64, 16],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3]),
                                                           MultiHeadUnionAdaIn(model_dim=self.model_dim,
                                                                               features_dims=[4 * 4, 4 * 8],
                                                                               heads=[16, 16],
                                                                               tensor_sizes=[16, 8],
                                                                               model_dim_out=self.model_dim,
                                                                               n_latent=num_latent,
                                                                               tensor_dims=[2, 3])] for _ in range(4)]))

        self.final = nn.Sequential(nn.Conv1d(in_channels=self.model_dim + 4,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   AdaIn1dUpd(self.model_dim, num_latent=num_latent),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=3, kernel_size=1))

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, noise, input, return_lattice=False):
        z, enc_lattices = self.encoder(input)
        z = z.reshape(-1, 1024)
        z = self.mapping(z)

        x = forward_style(self.start, noise, z)

        lattices_sizes = []

        for i in range(len(self.attentions_decoder)):
            x, lattice_size = self.attentions_decoder[i](x, z, noise[:, :3])
            lattices_sizes += lattice_size

        x = forward_style(self.final, torch.cat([x, noise], dim=1), z)

        lattices_sizes = enc_lattices + lattices_sizes

        return x.unsqueeze(2), lattices_sizes