from torch import nn
import torch
from layers.multihead_ct import MultiHeadUnion

def sum_hack(lists):
    start = lists[0]
    for i in range(1, len(lists)):
        start += lists[i]

    return start


class Model(nn.Module):
    def __init__(self, n_classes=13):
        super().__init__()

        self.n_classes = n_classes
        self.model_dim = 512

        self.first_process = nn.Sequential(nn.Conv1d(7, self.model_dim, kernel_size=1, bias=True),
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

        self.final = nn.Sequential(nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=self.model_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(self.model_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(in_channels=self.model_dim,
                                             out_channels=self.n_classes, kernel_size=1))

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, points, pts_pad, features):
        input_pts = points.permute(0, 2, 1)

        pcd = torch.cat([input_pts, features], dim=1)

        x = self.first_process(pcd)

        lattices_sizes = []
        for i in range(len(self.attentions_encoder)):
            x, lattice_size = self.attentions_encoder[i](x, (input_pts, pts_pad))

            lattices_sizes += lattice_size

        res = self.final(x)

        assert(res.size(-1) == pcd.size(-1))
        assert(res.size(1) == self.n_classes)

        return res
