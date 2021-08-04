from torch import nn
from torch.autograd import Function
import torch

from chamfer_extension import chamfer


# Chamfer's distance module @thibaultgroueix
# GPU tensors only
class ChamferFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert(xyz1.device == xyz2.device)
        assert(xyz1.size(0) == xyz2.size(0))
        assert(xyz1.size(2) == 3 and xyz2.size(2) == 3)

        assert(xyz1.size(-1) == xyz2.size(-1))
        assert(xyz1.is_contiguous() and xyz2.is_contiguous())

        device = xyz1.device

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        dist1 = dist1.to(device)
        dist2 = dist2.to(device)
        idx1 = idx1.to(device)
        idx2 = idx2.to(device)

        chamfer.forward(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, graddist1, graddist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        device = xyz1.device

        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()

        gradxyz1 = torch.zeros(xyz1.size())
        gradxyz2 = torch.zeros(xyz2.size())

        gradxyz1 = gradxyz1.to(device)
        gradxyz2 = gradxyz2.to(device)

        chamfer.backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        return gradxyz1, gradxyz2


class ChamferDist(nn.Module):
    def __init__(self):
        super(ChamferDist, self).__init__()

    def forward(self, input1, input2):
        return ChamferFunction.apply(input1, input2)


def loss_chamfer(pc_1, pc_2):
    # in 3d case
    chamfer_dist = ChamferDist()

    dist_1, dist_2 = chamfer_dist(pc_1[:, :, 0].permute(0, 2, 1).contiguous(),
                                  pc_2[:, :, 0].permute(0, 2, 1).contiguous())

    loss = torch.mean(dist_1) + torch.mean(dist_2)

    return loss


# loss as in pcn
def loss_chamfer_adj(pc_1, pc_2):
    # in 3d case
    chamfer_dist = ChamferDist()

    dist_1, dist_2 = chamfer_dist(pc_1[:, :, 0].permute(0, 2, 1).contiguous(),
                                  pc_2[:, :, 0].permute(0, 2, 1).contiguous())

    loss = torch.mean(torch.sqrt(dist_1)) + torch.mean(torch.sqrt(dist_2))

    return loss / 2


def loss_chamder_2d(pc_1, pc_2):
    # in 2d case

    zeros_1 = torch.zeros(pc_1.size(0), 1, 1, pc_1.size(-1)).to(device=pc_1.device)
    zeros_2 = torch.zeros(pc_2.size(0), 1, 1, pc_2.size(-1)).to(device=pc_1.device)

    return loss_chamfer(torch.cat([pc_1, zeros_1], dim=1), torch.cat([pc_2, zeros_2], dim=1))
