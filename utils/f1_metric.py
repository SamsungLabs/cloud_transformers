# reference https://github.com/lmb-freiburg/what3d
import open3d
import typing
import torch

import numpy as np


def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float = 0.01) -> typing.Tuple[
    float, float, float]:
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    # d1 = open3d.compute_point_cloud_to_point_cloud_distance(gt, pr)
    # d2 = open3d.compute_point_cloud_to_point_cloud_distance(pr, gt)
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    return fscore, precision, recall


def get_f1_scores(pcd, pcd_gt, th=0.01):
    assert (pcd.shape[0] == pcd_gt.shape[0])
    batch_num = pcd.shape[0]

    pcd_np = pcd.detach().cpu().numpy()
    pcd_gt_np = pcd_gt.detach().cpu().numpy()

    fs, ps, rs = [], [], []

    for b in range(batch_num):
        point_cloud_gt = open3d.geometry.PointCloud()
        point_cloud_gt.points = open3d.utility.Vector3dVector(pcd_gt_np[b].T)

        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(pcd_np[b].T)

        fscore, precision, recall = calculate_fscore(point_cloud_gt, point_cloud, th)

        fs.append(fscore)
        ps.append(precision)
        rs.append(recall)
    return fs, ps, rs


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


def get_f1_scores_merge(pcd, pcd_2, pcd_gt, th=0.01):
    assert (pcd.shape[0] == pcd_gt.shape[0])
    assert (pcd.shape[0] == pcd_2.shape[0])

    batch_num = pcd.shape[0]

    pcd_np = torch.cat([pcd, pcd_2], dim=-1).detach().cpu().numpy()
    pcd_gt_np = pcd_gt.detach().cpu().numpy()

    fs, ps, rs = [], [], []

    for b in range(batch_num):

        point_cloud_gt = open3d.geometry.PointCloud()
        point_cloud_gt.points = open3d.utility.Vector3dVector(pcd_gt_np[b].T)

        point_cloud = open3d.geometry.PointCloud()
        # print(resample_pcd(pcd_np[b].T, pcd_gt.shape[-1]).shape)
        point_cloud.points = open3d.utility.Vector3dVector(resample_pcd(pcd_np[b].T, pcd_gt.shape[-1]))

        fscore, precision, recall = calculate_fscore(point_cloud_gt, point_cloud, th)

        fs.append(fscore)
        ps.append(precision)
        rs.append(recall)
    return fs, ps, rs
