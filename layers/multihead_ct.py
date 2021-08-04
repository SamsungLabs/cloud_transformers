import torch
from torch import nn

from layers.cloud_transform import Splat, Slice, DifferentiablePositions

from layers.utils import PlaneTransformer, VolTransformer


class MultiHead(nn.Module):
    def __init__(self,
                 model_dim,
                 in_feature_dim,
                 out_model_dim,
                 tensor_size,
                 tensor_dim,
                 heads,
                 scales=False):
        super().__init__()

        assert(tensor_dim == 3 or tensor_dim == 2)

        self.in_feature_dim = in_feature_dim
        self.out_model_dim = out_model_dim

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

        self.slice = Slice(tensor_size=self.tensor_size,
                           dim=self.tensor_dim,
                           heads=self.heads)

        if self.tensor_dim == 3:
            self.conv = nn.Sequential(nn.Conv3d(self.heads * self.in_feature_dim,
                                                self.heads * self.in_feature_dim,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                groups=self.heads,
                                                bias=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(self.heads * self.in_feature_dim,
                                                self.heads * self.in_feature_dim,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1,
                                                groups=self.heads,
                                                bias=True))

        self.after = nn.Sequential(nn.BatchNorm1d(self.heads * self.in_feature_dim),
                                   nn.ReLU(inplace=True))

        self.transform = None

        if self.tensor_dim == 3:
            self.transform = VolTransformer(self.heads, scales=scales)
        else:
            self.transform = PlaneTransformer(self.heads, scales=scales)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.zeros_(self.key_bn.weight)

    def forward(self, input, orig_pcd, return_lattice=False):
        # input of size (b, c, p)
        if isinstance(orig_pcd, tuple):
            orig_pcd, pts_padd = orig_pcd
        else:
            pts_padd = None

        key_values = self.keys_values_pred(input)
        keys_res = self.key_bn(key_values[:, :self.heads * 3])
        values = self.values_bn(key_values[:, self.heads * 3:])

        keys = self.transform((orig_pcd[:, None] + keys_res.reshape(input.shape[0], self.heads, 3, input.shape[-1])))
        keys = keys.reshape(input.shape[0], self.heads * self.tensor_dim, input.shape[-1])

        assert (keys.shape[1] == self.heads * self.tensor_dim)
        lattice = torch.tanh(keys)

        local_coord, flattened_index = self.diff_poss(lattice)

        z = self.splat(local_coord, flattened_index, values, pts_padd)

        # collect occupancy stats
        with torch.no_grad():
            occ = (torch.abs(z) > 1e-9).sum().float() / (keys.size(0) * self.in_feature_dim * self.heads)

        result = self.after(self.slice(local_coord, flattened_index, self.conv(z), pts_padd))

        with torch.no_grad():
            stats = (occ,
                     torch.mean(keys).detach(),
                     torch.var(keys).detach(),
                     None)

        if return_lattice:
            result = result, lattice

        return result, stats


class MultiHeadUnion(nn.Module):
    def __init__(self,
                 model_dim,
                 features_dims,
                 tensor_sizes,
                 tensor_dims,
                 heads,
                 model_dim_out=None,
                 scales=False):
        super().__init__()
        self.model_dim = model_dim

        self.features_dims = features_dims
        self.tensor_sizes = tensor_sizes
        self.tensor_dims = tensor_dims
        self.heads = heads

        assert(len(self.features_dims) == len(self.tensor_sizes))
        assert(len(self.features_dims) == len(self.tensor_dims))
        assert(len(self.features_dims) == len(self.heads))

        self.model_dim_out = model_dim_out

        if self.model_dim_out is None:
            self.model_dim_out = self.model_dim

        self.prenorm = nn.Sequential()

        self.after = nn.Sequential(nn.Conv1d(sum([h*f for h, f in zip(self.heads, self.features_dims)]),
                                             self.model_dim_out,
                                             kernel_size=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm1d(self.model_dim_out),
                                   nn.ReLU(inplace=True))

        self.shortcut = nn.Sequential()
        if self.model_dim != self.model_dim_out:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv1d(self.model_dim, self.model_dim_out,
                                               kernel_size=1, stride=1, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm1d(self.model_dim_out))

        attentions = []
        for feature_dim, tensor_size, tensor_dim, head in zip(self.features_dims,
                                                              self.tensor_sizes,
                                                              self.tensor_dims,
                                                              self.heads):
            attentions.append(MultiHead(model_dim=self.model_dim,
                                        in_feature_dim=feature_dim,
                                        out_model_dim=self.model_dim_out,
                                        tensor_size=tensor_size,
                                        tensor_dim=tensor_dim,
                                        heads=head,
                                        scales=scales))

        self.attentions = nn.ModuleList(attentions)

        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(self, x, orig_pcd):
        # input of size (b, c, p)

        results = []
        stats = []

        x = self.prenorm(x)
        residual = self.shortcut(x)

        for attention in self.attentions:
            head_result, stat = attention(x, orig_pcd)
            results.append(head_result)
            stats.append(stat)

        gathered = self.after(torch.cat(results, dim=1))

        return residual + gathered, stats
