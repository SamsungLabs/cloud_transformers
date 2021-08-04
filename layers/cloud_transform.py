# this is an implementation of core operations for the Cloud Transformers paper (https://arxiv.org/abs/2007.11679)

import numpy as np

import torch
from torch import nn
from torch_scatter import scatter_max

from layers.utils import trilinear_coords, bilinear_coords


class GradientBalancing(torch.autograd.Function):
    """
    Decreases gradient response by scale during the backpropagation.
    See section 3.2 for the discussion
    """
    @staticmethod
    def forward(ctx, input, scale):
        return input * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


balance_op = GradientBalancing.apply


class DifferentiableGridModule(nn.Module):
    def __init__(self, tensor_size=20, heads=4, dim=3):
        '''
        :param tensor_size: spatial resolution of the feature map
                            tuple with len() == dim or int
        :param heads: The num of parallel de/rasterizations. Int > 0
        :param dim: the dimension of feature tensor to de/rasterize. Either 2 or 3
        '''
        super().__init__()
        self.dim = dim
        self.heads = heads

        if isinstance(tensor_size, int):
            self.tensor_size = dim * [tensor_size]
        else:
            assert (isinstance(tensor_size, tuple))
            assert (len(tensor_size) == dim)
            self.tensor_size = tensor_size

        tensor_mod = torch.tensor(self.tensor_size,
                                  dtype=torch.float32,
                                  requires_grad=False)[None, :, None]
        self.register_buffer('tensor_mod', tensor_mod)

        # feature map cell vertices amount
        if self.dim == 3:
            self.spread_size = 8
        else:
            self.spread_size = 4

        self.eps = 1e-7


class DifferentiablePositions(DifferentiableGridModule):
    '''
        A module which computes bi/tri-linear coordinates of each point wrt the enclosing feature map cell
        The de/rasterization happens using either [tensor_size, tensor_size]
        or [tensor_size, tensor_size, tensor_size] grid.
    '''

    def __init__(self, tensor_size=20, heads=4, dim=3):
        super().__init__(tensor_size, heads, dim)

    def forward(self, keys):
        '''
        :param keys: points' spatial locations to differentiably rasterize/de-rasterize
              torch float32 tensor of shape [batch_size, heads * dim, num_points]

        :return: local_coordinate: bi/tri-linear coordinates of each point wrt the enclosing feature map cell
                 torch float32 tensor of shape [batch_size, heads, num_cell_vertices, num_points]

                 flattened_index: integer indices of the corresponding cell vertices.
                 torch long tensor of shape [batch_size, heads, num_cell_vertices, num_points]
        '''

        assert (keys.size(1) == self.heads * self.dim)

        batch_size = keys.size(0)
        num_points = keys.size(-1)

        keys = keys.reshape(batch_size * self.heads, self.dim, keys.size(-1))

        keys = keys.clamp(-1 + self.eps, 1 - self.eps)

        # increase scale, but keep the gradients the same
        keys_scaled = balance_op((keys + 1.0), (self.tensor_mod - 1) * 0.5)

        if self.dim == 3:
            local_coordinate, actual_d_keys = trilinear_coords(keys_scaled)
        else:
            local_coordinate, actual_d_keys = bilinear_coords(keys_scaled)

        assert ((actual_d_keys < self.tensor_mod.long()).all())
        assert ((actual_d_keys > -1).all())

        local_coordinate = local_coordinate.reshape(batch_size, self.heads,
                                                    self.spread_size, num_points)

        actual_d_keys = actual_d_keys.reshape(batch_size, self.heads,
                                              self.spread_size, self.dim, num_points)

        assert ((actual_d_keys < self.tensor_mod.long()).all())
        assert ((actual_d_keys > -1).all())

        if self.dim == 3:
            flattened_index = actual_d_keys[:, :, :, 0] * self.tensor_size[1] * self.tensor_size[2] + \
                              actual_d_keys[:, :, :, 1] * self.tensor_size[2] + actual_d_keys[:, :, :, 2]

        else:
            flattened_index = actual_d_keys[:, :, :, 0] * self.tensor_size[1] + \
                              actual_d_keys[:, :, :, 1]

        return local_coordinate, flattened_index


class Splat(DifferentiableGridModule):
    '''
        A module which performs diffirentiable rasterization (splatting) into 2D/3D feature grid.
    '''
    def __init__(self, tensor_size=20, heads=4, dim=3):
        super().__init__(tensor_size, heads, dim)

    def forward(self, local_coordinate, flattened_index, features, pts_padding=None):
        '''
        :param local_coordinate: output of DifferentiablePositions
               torch float32 tensor of shape [batch_size, heads, num_cell_vertices, num_points]

        :param flattened_index: output of DifferentiablePositions
               torch long tensor of shape [batch_size, heads, num_cell_vertices, num_points]

        :param features: points' features
               torch float32 tensor of shape [batch_size, heads * feature_dim, num_points]

        :return:  feature map of shape [batch_size, heads * feature_dim, W, W] if
                  dim = 2 or [batch_size, heads * feature_dim, W, W, W] if dim = 3
                  W = tensor_size
                  (note: code also works for non-cubic / non-square feature maps)
        '''
        assert (features.dtype == torch.float32)
        assert (features.size(1) % self.heads == 0)

        feature_dim = features.size(1) // self.heads

        batch_size = features.size(0)
        num_points = features.size(-1)

        # pre-splat here with "local coordinates"
        features = features.reshape(batch_size, self.heads, feature_dim, num_points)

        if pts_padding is not None:
            features = features * pts_padding[:, None, None, :]

        pre_splat = (features[:, :, :, None] * local_coordinate[:, :, None])

        # create 3d/2d map where we rasterize
        z_r = torch.zeros((batch_size,
                           self.heads,
                           feature_dim,
                           int(np.prod(self.tensor_size))), device=pre_splat.device)

        flattened_index = flattened_index[:, :, None]

        z, _ = scatter_max(out=z_r, dim=3,
                           index=flattened_index.reshape(batch_size, self.heads, 1, -1),
                           src=pre_splat.reshape(batch_size, self.heads, feature_dim, -1))

        z = z.reshape(
            (batch_size,
             self.heads * feature_dim,
             *self.tensor_size))

        return z


class Slice(DifferentiableGridModule):
    '''
        A module which performs differentiable sampling of 2D/3D feature grid.
    '''
    def __init__(self, tensor_size=20, heads=4, dim=3):
        super().__init__(tensor_size, heads, dim)

    def forward(self, local_coordinate, flattened_index, convolved, pts_padding=None):
        '''
        :param local_coordinate: output of DifferentiablePositions
                torch float32 tensor of shape [batch_size, heads, num_cell_vertices, num_points]

        :param flattened_index: output of DifferentiablePositions
                torch long tensor of shape [batch_size, heads, num_cell_vertices, num_points]

        :param convolved: feature map of shape [batch_size, heads * feature_dim, W, W] if
                          dim = 2 or [batch_size, heads * feature_dim, W, W, W] if dim = 3
                W = tensor_size
                (note: code also works for non-cubic / non-square feature maps)

        :return: sliced features
                 torch float32 tensor of shape [batch_size, heads * feature_dim, num_points]
        '''
        assert (convolved.size(1) % self.heads == 0)
        feature_dim = convolved.size(1) // self.heads

        batch_size = local_coordinate.size(0)
        num_points = local_coordinate.size(-1)

        flattened_index = flattened_index[:, :, None].expand(-1, -1, feature_dim, -1, -1)

        z_rr = convolved.reshape(batch_size, self.heads, feature_dim, -1)

        gathered = torch.gather(input=z_rr,
                                dim=3,
                                index=flattened_index.reshape(batch_size, self.heads, feature_dim, -1))
        gathered = gathered.reshape(batch_size, self.heads, feature_dim, self.spread_size, num_points)

        sliced = (gathered * local_coordinate[:, :, None]).sum(dim=3)
        sliced = sliced.reshape(batch_size, self.heads * feature_dim, num_points)

        if pts_padding is not None:
            sliced = sliced * pts_padding[:, None, :]

        return sliced
