import torch
import numpy as np


def sphere_noise(batch, num_pts, device):
    with torch.no_grad():
        theta = 2 * np.pi * torch.rand(batch, num_pts, device=device)
        phi = torch.acos(1 - 2 * torch.rand(batch, num_pts, device=device))
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

    return torch.stack([x, y, z], dim=1)


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


def partial_postproces(partial_pcd, gt_size):
    with torch.no_grad():
        input_pcd_sizes = partial_pcd.shape[1]

        new_parts = []
        new_withnoise = []

        for batch_id in range(partial_pcd.shape[0]):
            current_pcd = partial_pcd[batch_id]
            valid_mask = ~(current_pcd == 0.0).all(dim=1)

            without_zeros = current_pcd[valid_mask]

            noise_padd = sphere_noise(1, gt_size - without_zeros.shape[0], torch.device('cpu'))[0].permute(1, 0)

            labels_part = torch.ones(without_zeros.shape[0], 1)
            labels_noise = torch.zeros(noise_padd.shape[0], 1)

            noise_withlabels = torch.cat([noise_padd, labels_noise], dim=1)
            part_withlabels = torch.cat([without_zeros, labels_part], dim=1)

            labeled_noise = torch.cat([noise_withlabels, part_withlabels], dim=0)

            without_zeros = resample_pcd(without_zeros, input_pcd_sizes)
            new_parts.append(without_zeros)
            new_withnoise.append(labeled_noise)

    return torch.stack(new_parts, dim=0), torch.stack(new_withnoise, dim=0)