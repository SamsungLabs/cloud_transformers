# reference https://github.com/erikwijmans/Pointnet2_PyTorch
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py

import pathlib


import random

import scipy
import scipy.ndimage
import scipy.interpolate
from scipy.linalg import expm, norm


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, label):
        for t in self.transforms:
            data, label = t(data, label)
        return data, label


def M_x(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


class RandomRotate(object):
    def __init__(self, rotate_angle=None, along_z=True):
        self.rotate_angle = rotate_angle
        self.along_z = along_z

    def __call__(self, data):
        if self.rotate_angle is None:
            rotate_angle = np.random.uniform() * 2 * np.pi
        else:
            rotate_angle = self.rotate_angle
        cosval, sinval = np.cos(rotate_angle), np.sin(rotate_angle)
        if self.along_z:
            rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
        else:
            rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])

        data[:, 0:3] = np.dot(data[:, 0:3], rotation_matrix.T)

        # if data.shape[1] > 3:  # use normal
        #     data[:, 6:9] = np.dot(data[:, 6:9], rotation_matrix)
        return data


class RandomRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data):
        angles = np.clip(self.angle_sigma*np.random.randn(3), -self.angle_clip, self.angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        data[:, 0:3] = np.dot(data[:, 0:3], R)
        # if data.shape[1] > 3:  # use normal
        #     data[:, 3:6] = np.dot(data[:, 3:6], R)
        return data


class RandomSymmetries(object):
    def __init__(self, do_sym=(True, False, False)):
        self.do_sym = do_sym
        assert(len(self.do_sym) == 3)

    def __call__(self, data):
        scaler = []
        for flag in self.do_sym:
            if flag:
                scaler.append(np.round(np.random.uniform()) * 2 - 1)
            else:
                scaler.append(1)
        data[:, 0:3] *= np.stack(scaler, axis=0).astype(np.float32)
        return data


class RandomRotateV2(object):
    def __init__(self, rotation_augmentation_bound=((-np.pi / 32, np.pi / 32), (-np.pi / 32, np.pi / 32), (-np.pi, np.pi))):
        self.rotation_augmentation_bound = rotation_augmentation_bound

    def __call__(self, data):
        rot_mats = []
        for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound):
            theta = 0
            axis = np.zeros(3)
            axis[axis_ind] = 1
            if rot_bound is not None:
                theta = np.random.uniform(*rot_bound)
            rot_mats.append(M_x(axis, theta))
        # Use random order
        np.random.shuffle(rot_mats)
        rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]

        data[:, 0:3] = np.dot(data[:, 0:3], rot_mat)

        return data


class RandomScale(object):
    def __init__(self, scale_low=0.8, scale_high=1.2, anisotropic=True):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.anisotropic = anisotropic

    def __call__(self, data):
        if self.anisotropic:
            scale = np.random.uniform(self.scale_low, self.scale_high, size=(3))
        else:
            scale = np.random.uniform(self.scale_low, self.scale_high)

        data[:, 0:3] *= scale
        return data


class ZeroShift(object):
    def __init__(self):
        pass

    def __call__(self, data):
        shift = np.mean(data[:, 0:3], axis=0)
        data[:, 0:3] -= shift
        return data


class RandomShift(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data[:, 0:3] += shift
        return data


class RandomShiftLast(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 1)
        data[:, 6:7] += shift
        return data


class RandomJitterLast(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 1), -1 * self.clip, self.clip)
        data[:, 6:7] += jitter
        return data


class RandomColorShift(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data[:, 3:6] += shift
        return data


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 3), -1 * self.clip, self.clip)
        data[:, 0:3] += jitter
        return data


class RandomXSymmetry(object):
    def __init__(self):
        pass

    def __call__(self, data):
        sym = np.array([np.random.choice([0, 1]) * 2 - 1, 1.0, 1.0]).astype(np.float32)
        data[:, 0:3] *= sym
        return data


class RandomColorDrop(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, data):
        drop = np.random.choice([0, 1], p=[self.p, 1 - self.p]).astype(np.float32)
        data[:, 3:6] *= drop
        return data


class RandomJitterColor(object):
    def __init__(self, sigma=0.02, clip=0.1):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 3), -1 * self.clip, self.clip)
        data[:, 3:6] += jitter
        return data


class RandomJitterNorm(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 3), -1 * self.clip, self.clip)
        data[:, 6:9] += jitter
        return data


class RandomShiftNorm(object):
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = np.random.uniform(-self.shift_range, self.shift_range, 3)
        data[:, 6:9] += shift
        return data

### this part has been taken from https://github.com/chrischoy/SpatioTemporalSegmentation/

class ChromaticTranslation(object):
    """Add random color to the image, input must be an array in [0,255] or a PIL image"""

    def __init__(self, trans_range_ratio=1e-1):
        """
        trans_range_ratio: ratio of translation i.e. 255 * 2 * ratio * rand(-0.5, 0.5)
        """
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, data):
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 2 * self.trans_range_ratio
            data[:, 3:6] = np.clip(tr + data[:, 3:6], 0, 1.0)
        return data


class ChromaticAutoContrast(object):

    def __init__(self, randomize_blend_factor=True, blend_factor=0.5):
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, data):
        feats = data[:, 3:6]

        if random.random() < 0.2:
            # mean = np.mean(feats, 0, keepdims=True)
            # std = np.std(feats, 0, keepdims=True)
            # lo = mean - std
            # hi = mean + std
            lo = np.min(feats, 0, keepdims=True)
            hi = np.max(feats, 0, keepdims=True)

            scale = 1 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats = (1 - blend_factor) * feats + blend_factor * contrast_feats

            data[:, 3:6] = feats

        return data


class ChromaticJitter(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        if random.random() < 0.95:
            noise = np.random.randn(data.shape[0], 3)
            noise *= self.std
            data[:, 3:6] = np.clip(noise + data[:, 3:6], 0, 1)
        return data


class HueSaturationTranslation(object):

    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max, saturation_max):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data):
        feats = data[:, 3:6] * 255.0

        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        data[:, 3:6] = feats / 255.0

        return data


class ElasticDistortion:

    def __init__(self, distortion_params=((20, 100), (80, 320))):
        self.distortion_params = distortion_params

    def elastic_distortion(self, pointcloud, granularity, magnitude):
        """Apply elastic distortion on sparse coordinate space.
          pointcloud: numpy array of (number of points, at least 3 spatial dims)
          granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
          magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords = pointcloud[:, :3]
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min +
                                       granularity * (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        pointcloud[:, :3] = coords + interp(coords) * magnitude
        return pointcloud

    def __call__(self, data):
        pointcloud = data[:, :3]
        if self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    pointcloud = self.elastic_distortion(pointcloud, granularity, magnitude)

        data[:, :3] = pointcloud

        return data


class RandomHorizontalFlip(object):

    def __init__(self, upright_axis, is_temporal):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.is_temporal = is_temporal
        self.D = 4 if is_temporal else 3
        self.upright_axis = {'x': 0, 'y': 1, 'z': 2}[upright_axis.lower()]
        # Use the rest of axes for flipping.
        self.horz_axes = set(range(self.D)) - set([self.upright_axis])

    def __call__(self, coords, feats, labels):
        if random.random() < 0.95:
            for curr_ax in self.horz_axes:
                if random.random() < 0.5:
                    coord_max = np.max(coords[:, curr_ax])
                    coords[:, curr_ax] = coord_max - coords[:, curr_ax]
        return coords, feats, labels
### end


g_class2color = {'ceiling': [0,255,0],
                 'floor': [0,0,255],
         'wall':  [0,255,255],
         'beam':        [255,255,0],
         'column':      [255,0,255],
         'window':      [100,100,255],
         'door':        [200,200,100],
         'table':       [170,120,200],
         'chair':       [255,0,0],
         'sofa':        [200,100,100],
         'bookcase':    [10,200,100],
         'board':       [200,200,200],
         'clutter':     [50,50,50]}


class_order = ['ceiling', 'floor', 'wall', 'beam',
               'column', 'window', 'door', 'table',
               'chair', 'sofa', 'bookcase', 'board', 'clutter']


def convert_lables_to_color(labels_np):
    batch = []
    for b_id in range(labels_np.shape[0]):
        current_pcd = []
        for p_id in range(labels_np.shape[1]):
            current_color = g_class2color[class_order[labels_np[b_id][p_id]]]

            current_pcd.append(current_color)
        batch.append(current_pcd)

    return np.array(batch)


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip() for line in f]


def _load_data_file(name):
    f = h5py.File(name, 'r')
    data = f["data"][:]
    label = f["label"][:]
    return data, label


class Indoor3DSemSeg(data.Dataset):
    def __init__(self, data_dir, num_points,
                 train=True,
                 data_precent=1.0,
                 aug=False,
                 test_area='Area_5'):
        super().__init__()
        self.data_precent = data_precent
        self.data_dir = pathlib.Path(data_dir)

        self.aug = aug

        self.train, self.num_points = train, num_points

        all_files = _get_data_files(self.data_dir.joinpath("all_files.txt"))
        room_filelist = _get_data_files(self.data_dir.joinpath("room_filelist.txt"))

        data_batchlist, label_batchlist = [], []
        for f in all_files:
            data, label = _load_data_file(self.data_dir.joinpath(pathlib.Path(f).name))
            data_batchlist.append(data)
            label_batchlist.append(label)

        data_batches = np.concatenate(data_batchlist, 0)
        labels_batches = np.concatenate(label_batchlist, 0)

        self.test_area = test_area
        train_idxs, test_idxs = [], []
        for i, room_name in enumerate(room_filelist):
            if self.test_area in room_name:
                test_idxs.append(i)
            else:
                train_idxs.append(i)

        if self.train:
            self.points = data_batches[train_idxs, ...]
            self.labels = labels_batches[train_idxs, ...]

        else:
            self.points = data_batches[test_idxs, ...]
            self.labels = labels_batches[test_idxs, ...]

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)

        current_points_np = self.points[idx, pt_idxs, :6].copy()

        # current_points_np = ZeroShift()(current_points_np)

        # current_points_np_z = self.points[idx, pt_idxs, 2:3].copy()

        if self.aug:
            current_points_np = RandomRotate(along_z=True)(current_points_np)
            current_points_np = RandomScale(anisotropic=True)(current_points_np)
            current_points_np = RandomSymmetries()(current_points_np)
            current_points_np = RandomJitter()(current_points_np)

            # color part
            current_points_np = ChromaticAutoContrast()(current_points_np)
            current_points_np = ChromaticTranslation(0.10)(current_points_np)
            current_points_np = ChromaticJitter(0.05)(current_points_np)
            current_points_np = HueSaturationTranslation(0.5, 0.20)(current_points_np)

        current_points = torch.from_numpy(current_points_np).type(
            torch.FloatTensor
        )
        current_labels = torch.from_numpy(self.labels[idx, pt_idxs].copy()).type(
            torch.LongTensor
        )

        return current_points, current_labels

    def __len__(self):
        return int(self.points.shape[0] * self.data_precent)

    def set_num_points(self, pts):
        self.num_points = pts

    def randomize(self):
        pass


if __name__ == "__main__":
    dset = Indoor3DSemSeg(16, "./", train=True)
    print(dset[0])
    print(len(dset))
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
    for i, data in enumerate(dloader, 0):
        inputs, labels = data
        if i == len(dloader) - 1:
            print(inputs.size())
