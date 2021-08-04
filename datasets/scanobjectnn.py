import numpy as np

import torch
from torch.utils.data import Dataset

import h5py


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


def center_data(pcs):
    for pc in pcs:
        centroid = np.mean(pc, axis=0)
        pc[:,0]-=centroid[0]
        pc[:,1]-=centroid[1]
        pc[:,2]-=centroid[2]
    return pcs


def normalize_data(pcs):
    for pc in pcs:
        #get furthest point distance then normalize
        d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
        pc /= d

        # pc[:,0]/=max(abs(pc[:,0]))
        # pc[:,1]/=max(abs(pc[:,1]))
        # pc[:,2]/=max(abs(pc[:,2]))
    return pcs


def load_withmask_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    mask = f['mask'][:]

    return data, label, mask


def convert_to_binary_mask(masks):
    binary_masks = []
    for i in range(masks.shape[0]):
        binary_mask = np.ones(masks[i].shape)
        bg_idx = np.where(masks[i, :] == -1)
        binary_mask[bg_idx] = 0

        binary_masks.append(binary_mask)

    binary_masks = np.array(binary_masks)
    return binary_masks


class ScanObjectNN(Dataset):
    def __init__(self, data_dir, center=True, normalize=True, train=False, subsample=None):
        self.data, self.label, self.mask = load_withmask_h5(data_dir)
        self.mask = convert_to_binary_mask(self.mask)

        if center:
            self.data = center_data(self.data)

        if normalize:
            self.data = normalize_data(self.data)

        self.train = train

        self.subsample = subsample

    def __getitem__(self, item):
        pointcloud = self.data[item][None]
        label = self.label[item]
        mask = self.mask[item]

        if self.train:
            pointcloud = jitter_point_cloud(pointcloud)
            pointcloud = rotate_point_cloud(pointcloud)

        pc_np = pointcloud[0].copy()
        ma_np = mask.copy()

        if self.subsample is not None:
            idx = np.random.choice(pc_np.shape[0], size=self.subsample, replace=False)
            pc_np = pc_np[idx]
            ma_np = ma_np[idx]

        pc = torch.from_numpy(pc_np).type(torch.FloatTensor)
        ma = torch.from_numpy(ma_np).type(torch.LongTensor)

        return pc, label, ma

    def __len__(self):
        return self.data.shape[0]
