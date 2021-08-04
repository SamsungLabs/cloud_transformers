import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from layers.cloud_transform import Splat, Slice, DifferentiablePositions
from layers.utils import VolTransformer, PlaneTransformer


class MultiHeadPool(nn.Module):
    def __init__(self,
                 model_dim,
                 in_feature_dim,
                 tensor_size,
                 tensor_dim,
                 heads,
                 scales=False):
        super().__init__()

        assert(tensor_dim == 3 or tensor_dim == 2)

        self.in_feature_dim = in_feature_dim

        self.model_dim = model_dim
        self.tensor_size = tensor_size
        self.tensor_dim = tensor_dim

        self.heads = heads

        self.keys_values_pred = nn.Sequential(nn.Conv1d(self.model_dim,
                                                        self.heads * (self.in_feature_dim + 3),
                                                        kernel_size=1, bias=False))

        self.values_bn = nn.BatchNorm1d(self.heads * self.in_feature_dim)
        self.key_bn = nn.BatchNorm1d(self.heads * 3)

        self.diff_poss = DifferentiablePositions(tensor_size=self.tensor_size,
                                                 dim=self.tensor_dim,
                                                 heads=self.heads)

        self.splat = Splat(tensor_size=self.tensor_size,
                           dim=self.tensor_dim,
                           heads=self.heads)
        self.transform = None

        if self.tensor_dim == 3:
            self.transform = VolTransformer(self.heads, scales=scales)
        else:
            self.transform = PlaneTransformer(self.heads, scales=scales)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.zeros_(self.key_bn.weight)

    def forward(self, input, orig_pcd, return_lattice=False):
        key_values = self.keys_values_pred(input)

        keys_res = self.key_bn(key_values[:, :self.heads * 3])
        values = self.values_bn(key_values[:, self.heads * 3:])

        keys = self.transform((orig_pcd[:, None] + keys_res.reshape(input.shape[0], self.heads, 3, input.shape[-1])))
        keys = keys.reshape(input.shape[0], self.heads * self.tensor_dim, input.shape[-1])

        assert (keys.shape[1] == self.heads * self.tensor_dim)

        lattice = torch.tanh(keys)

        local_coord, flattened_index = self.diff_poss(lattice)

        z = self.splat(local_coord, flattened_index, values)

        with torch.no_grad():
            occ = (torch.abs(z) > 1e-9).sum().float() / (keys.size(0) * self.in_feature_dim * self.heads)

        result = z

        with torch.no_grad():
            stats = (occ,
                     torch.mean(keys).detach(),
                     torch.var(keys).detach(),
                     None)

        if return_lattice:
            result = result, lattice

        return result, stats
