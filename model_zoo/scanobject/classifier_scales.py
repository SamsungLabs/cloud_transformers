import torch
from torch import nn

from layers.multihead_ct import MultiHeadUnion
from layers.multihead_ct_pool import MultiHeadPool


from layers.v2v_groups import Res3DBlock, Pool3DBlock, Basic3DBlock
from unet2d.unet_parts import Res2DBlock, Basic2DBlock

def sum_hack(lists):
    start = lists[0]
    for i in range(1, len(lists)):
        start += lists[i]

    return start


class ReLUDropoutInplace(torch.nn.Module):
    def __init__(self, p : float):
        super(ReLUDropoutInplace, self).__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            p1m = 1. - self.p
            mask = torch.rand_like(input) < p1m
            mask *= (input > 0)
            return input.masked_fill_(~mask, 0).mul_(1. / p1m)
        else:
            return input.clamp_(min = 0)


class Model(nn.Module):
    def __init__(self, n_classes=15):
        super().__init__()

        self.n_classes = n_classes
        self.model_dim = 512

        self.first_process = nn.Sequential(nn.Conv1d(3, self.model_dim, kernel_size=1, bias=False),
                                           nn.BatchNorm1d(self.model_dim),
                                           nn.ReLU(inplace=True))

        # self.num_layers = 16

        self.attentions_encoder = nn.ModuleList(sum_hack([[MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4, 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[128, 32],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3], scales=True),
                                                 MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 4],
                                                                heads=[16, 16],
                                                                tensor_sizes=[64, 16],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3], scales=True),
                                                 MultiHeadUnion(model_dim=self.model_dim,
                                                                features_dims=[4 * 4, 4 * 8],
                                                                heads=[16, 16],
                                                                tensor_sizes=[16, 8],
                                                                model_dim_out=self.model_dim,
                                                                tensor_dims=[2, 3], scales=True)] for _ in range(4)]))

        self.pool3d = MultiHeadPool(model_dim=self.model_dim,
                                    in_feature_dim=4 * 8,
                                    heads=16,
                                    tensor_size=8,
                                    tensor_dim=3, scales=True)

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
                                    tensor_dim=2, scales=True)

        self.after_pool2d = nn.Sequential(Res2DBlock(16 * self.heads_2d, 32 * self.heads_2d, groups=16),
                                          nn.MaxPool2d(2),
                                          Res2DBlock(32 * self.heads_2d, 64 * self.heads_2d, groups=16),
                                          nn.MaxPool2d(2),
                                          Res2DBlock(64 * self.heads_2d, 64 * self.heads_2d, groups=16),
                                          nn.AdaptiveAvgPool2d((1, 1)))

        self.class_vector = nn.Sequential(nn.Linear((1024 + 1024), 1024),
                                          nn.BatchNorm1d(1024),
                                          nn.ReLU(inplace=True))

        self.class_head = nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1024, n_classes))

        self.mask_head = nn.Sequential(nn.Dropout(0.5),
                                       nn.Conv1d(in_channels=self.model_dim + 1024,
                                                 out_channels=256, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(256),
                                       ReLUDropoutInplace(0.5),
                                       nn.Conv1d(in_channels=256,
                                                 out_channels=1, kernel_size=1))

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, input, return_lattice=False):
        input = input.squeeze(2)
        orig = input

        x = self.first_process(input)

        # unet_skip = []
        lattices_sizes = []
        for i in range(len(self.attentions_encoder)):
            # unet_skip.append(x)
            x, lattice_size = self.attentions_encoder[i](x, orig)

            # x = self.attentions_encoder_ff[i](x)

            lattices_sizes += lattice_size

        res = x

        assert(res.size(-1) == input.size(-1))
        # assert(res.size(1) == self.n_classes)

        # class_pred = self.class_head(torch.max(res, dim=-1, keepdim=False)[0])
        to_3d, lattice_size_2d = self.pool3d(res, orig)
        to_2d, lattice_size_3d = self.pool2d(res, orig)

        lattices_sizes += [lattice_size_2d]
        lattices_sizes += [lattice_size_3d]

        pooled_2d = self.after_pool2d(to_2d).reshape(-1, 1024)
        pooled_3d = self.after_pool3d(to_3d).reshape(-1, 1024)

        class_vect = self.class_vector(torch.cat([pooled_2d, pooled_3d], dim=-1))
        class_pred = self.class_head(class_vect)

        mask_pred = self.mask_head(torch.cat([res, class_vect[:, :, None].expand(-1, -1, res.size(-1))], dim=1))

        mask_pred = mask_pred.unsqueeze(2)

        assert(class_pred.size(1) == self.n_classes)

        return class_pred, mask_pred, lattices_sizes
