import json
import logging
import random
import torch.utils.data.dataset


from enum import Enum, unique
from tqdm import tqdm


import h5py
# import pyexr
import open3d
import os

import cv2
import math
import numpy as np
import torch
import transforms3d
from io import BytesIO

# References: http://confluence.sensetime.com/pages/viewpage.action?pageId=44650315
# from config import cfg
# sys.path.append(cfg.MEMCACHED.LIBRARY_PATH)

# mc_client = None
# if cfg.MEMCACHED.ENABLED:
#     import mc
#     mc_client = mc.MemcachedClient.GetInstance(cfg.MEMCACHED.SERVER_CONFIG, cfg.MEMCACHED.CLIENT_CONFIG)


class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.png', '.jpg']:
            return cls._read_img(file_path)
        elif file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.exr']:
            return cls._read_exr(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def put(cls, file_path, file_content):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.pcd']:
            return cls._write_pcd(file_path, file_content)
        elif file_extension in ['.h5']:
            return cls._write_h5(file_path, file_content)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_img(cls, file_path):
        return cv2.imread(file_path, cv2.IMREAD_UNCHANGED) / 255.
    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    # @classmethod
    # def _read_exr(cls, file_path):
    #     return 1.0 / pyexr.open(file_path).get("Depth.Z").astype(np.float32)

    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    @classmethod
    def _read_pcd(cls, file_path):
        # pc = open3d.io.read_point_cloud(file_path)
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        # Avoid overflow while gridding
        return f['data'][()] * 0.9

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _write_pcd(cls, file_path, file_content):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(file_content)
        open3d.io.write_point_cloud(file_path, pc)

    @classmethod
    def _write_h5(cls, file_path, file_content):
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('data', data=file_content)


class Compose(object):
    def __init__(self, transforms):
        self.transformers = []
        for tr in transforms:
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            rnd_value = np.random.uniform(0, 1)
            if transform.__class__ in [NormalizeObjectPose]:
                data = transform(data)
            else:
                for k, v in data.items():
                    if k in objects and k in data:
                        if transform.__class__ in [
                                RandomCrop, RandomFlip, RandomRotatePoints, RandomScalePoints, RandomMirrorPoints
                        ]:
                            data[k] = transform(v, rnd_value)
                        else:
                            data[k] = transform(v)

        return data


class ToTensor(object):
    def __init__(self, parameters):
        pass

    def __call__(self, arr):
        shape = arr.shape
        if len(shape) == 3:    # RGB/Depth Images
            arr = arr.transpose(2, 0, 1)

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return torch.from_numpy(arr.copy()).float()


class Normalize(object):
    def __init__(self, parameters):
        self.mean = parameters['mean']
        self.std = parameters['std']

    def __call__(self, arr):
        arr = arr.astype(np.float32)
        arr /= self.std
        arr -= self.mean

        return arr


class CenterCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * .5
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * .5
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomCrop(object):
    def __init__(self, parameters):
        self.img_size_h = parameters['img_size'][0]
        self.img_size_w = parameters['img_size'][1]
        self.crop_size_h = parameters['crop_size'][0]
        self.crop_size_w = parameters['crop_size'][1]

    def __call__(self, img, rnd_value):
        img_w, img_h, _ = img.shape
        x_left = (img_w - self.crop_size_w) * rnd_value
        x_right = x_left + self.crop_size_w
        y_top = (img_h - self.crop_size_h) * rnd_value
        y_bottom = y_top + self.crop_size_h

        # Crop the image
        img = cv2.resize(img[int(y_top):int(y_bottom), int(x_left):int(x_right)], (self.img_size_w, self.img_size_h))
        img = img[..., np.newaxis] if len(img.shape) == 2 else img

        return img


class RandomFlip(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img, rnd_value):
        if rnd_value > 0.5:
            img = np.fliplr(img)

        return img


class RandomPermuteRGB(object):
    def __init__(self, parameters):
        pass

    def __call__(self, img):
        rgb_permutation = np.random.permutation(3)
        return img[..., rgb_permutation]


class RandomBackground(object):
    def __init__(self, parameters):
        self.random_bg_color_range = parameters['bg_color']

    def __call__(self, img):
        img_h, img_w, img_c = img.shape
        if not img_c == 4:
            return img

        r, g, b = [
            np.random.randint(self.random_bg_color_range[i][0], self.random_bg_color_range[i][1] + 1) for i in range(3)
        ]
        alpha = (np.expand_dims(img[:, :, 3], axis=2) == 0).astype(np.float32)
        img = img[:, :, :3]
        bg_color = np.array([[[r, g, b]]]) / 255.
        img = alpha * bg_color + (1 - alpha) * img

        return img


class RandomSamplePoints(object):
    def __init__(self, parameters):
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud


class RandomClipPoints(object):
    def __init__(self, parameters):
        self.sigma = parameters['sigma'] if 'sigma' in parameters else 0.01
        self.clip = parameters['clip'] if 'clip' in parameters else 0.05

    def __call__(self, ptcloud):
        ptcloud += np.clip(self.sigma * np.random.randn(*ptcloud.shape), -self.clip, self.clip).astype(np.float32)
        return ptcloud


class RandomRotatePoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        angle = 2 * math.pi * rnd_value
        trfm_mat = np.dot(transforms3d.axangles.axangle2mat([0, 1, 0], angle), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class RandomScalePoints(object):
    def __init__(self, parameters):
        self.scale = parameters['scale']

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        scale = np.random.uniform(1.0 / self.scale * rnd_value, self.scale * rnd_value)
        trfm_mat = np.dot(transforms3d.zooms.zfdir2mat(scale), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class RandomMirrorPoints(object):
    def __init__(self, parameters):
        pass

    def __call__(self, ptcloud, rnd_value):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        if rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif rnd_value > 0.5 and rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        return ptcloud


class NormalizeObjectPose(object):
    def __init__(self, parameters):
        input_keys = parameters['input_keys']
        self.ptcloud_key = input_keys['ptcloud']
        self.bbox_key = input_keys['bbox']

    def __call__(self, data):
        ptcloud = data[self.ptcloud_key]
        bbox = data[self.bbox_key]

        # Calculate center, rotation and scale
        # References:
        # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale
        ptcloud = np.dot(ptcloud - center, rotation) / scale
        ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        data[self.ptcloud_key] = ptcloud
        return data


@unique
class DatasetSubset(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


def collate_fn(batch):
    taxonomy_ids = []
    model_ids = []
    data = {}

    for sample in batch:
        taxonomy_ids.append(sample[0])
        model_ids.append(sample[1])
        _data = sample[2]
        for k, v in _data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    for k, v in data.items():
        data[k] = torch.stack(v, 0)

    return taxonomy_ids, model_ids, data


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, options, file_list, transforms=None):
        self.options = options
        self.file_list = file_list
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        rand_idx = -1
        if 'n_renderings' in self.options:
            rand_idx = random.randint(0, self.options['n_renderings'] - 1) if self.options['shuffle'] else 0

        for ri in self.options['required_items']:
            file_path = sample['%s_path' % ri]
            if type(file_path) == list:
                file_path = file_path[rand_idx]

            data[ri] = IO.get(file_path).astype(np.float32)

        if self.transforms is not None:
            data = self.transforms(data)

        return sample['taxonomy_id'], sample['model_id'], data


class ShapeNetDataLoader(object):
    def __init__(self, category_file_path, partial_path, complete_path,
                 n_renders=1,
                 n_input=2048, n_output=16384):
        self.partial_path = partial_path
        self.complete_path = complete_path
        self.n_input = n_input
        self.n_output = n_output
        self.n_renders = n_renders

        self.category_file_path = category_file_path

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(self.category_file_path) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        n_renderings = self.n_renders if subset == DatasetSubset.TRAIN else 1
        file_list = self._get_file_list(self._get_subset(subset), n_renderings)
        transforms = self._get_transforms(subset)
        return Dataset({
            'required_items': ['partial_cloud', 'gtcloud'],
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, subset):
        if subset == DatasetSubset.TRAIN:
            return Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.n_input
                },
                'objects': ['partial_cloud']
            },
            {
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': self.n_output
                },
                'objects': ['gtcloud']
            },
            {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            if subset == DatasetSubset.TEST:
                return Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': self.n_input
                    },
                    'objects': ['partial_cloud']
                }, {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }])
            else:
                return Compose([{
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': self.n_input
                    },
                    'objects': ['partial_cloud']
                },
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {
                        'n_points': self.n_output
                    },
                    'objects': ['gtcloud']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial_cloud', 'gtcloud']
                }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path': [
                        self.partial_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gtcloud_path':
                    self.complete_path % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class ShapeNetCarsDataLoader(ShapeNetDataLoader):
    def __init__(self, cfg):
        super(ShapeNetCarsDataLoader, self).__init__(cfg)

        # Remove other categories except cars
        self.dataset_categories = [dc for dc in self.dataset_categories if dc['taxonomy_id'] == '02958343']


class Completion3DDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud'] if subset == DatasetSubset.TEST else ['partial_cloud', 'gtcloud']

        return Dataset({
            'required_items': required_items,
            'shuffle': subset == DatasetSubset.TRAIN
        }, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        if subset == DatasetSubset.TRAIN:
            return Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial_cloud', 'gtcloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])
        else:
            return Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': cfg.CONST.N_INPUT_POINTS
                },
                'objects': ['partial_cloud']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial_cloud', 'gtcloud']
            }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id':
                    dc['taxonomy_id'],
                    'model_id':
                    s,
                    'partial_cloud_path':
                    cfg.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                    'gtcloud_path':
                    cfg.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH % (subset, dc['taxonomy_id'], s),
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


class KittiDataLoader(object):
    def __init__(self, cfg):
        self.cfg = cfg

        # Load the dataset indexing file
        self.dataset_categories = []
        with open(cfg.DATASETS.KITTI.CATEGORY_FILE_PATH) as f:
            self.dataset_categories = json.loads(f.read())

    def get_dataset(self, subset):
        file_list = self._get_file_list(self.cfg, self._get_subset(subset))
        transforms = self._get_transforms(self.cfg, subset)
        required_items = ['partial_cloud', 'bounding_box']

        return Dataset({'required_items': required_items, 'shuffle': False}, file_list, transforms)

    def _get_transforms(self, cfg, subset):
        return Compose([{
            'callback': 'NormalizeObjectPose',
            'parameters': {
                'input_keys': {
                    'ptcloud': 'partial_cloud',
                    'bbox': 'bounding_box'
                }
            },
            'objects': ['partial_cloud', 'bounding_box']
        }, {
            'callback': 'RandomSamplePoints',
            'parameters': {
                'n_points': cfg.CONST.N_INPUT_POINTS
            },
            'objects': ['partial_cloud']
        }, {
            'callback': 'ToTensor',
            'objects': ['partial_cloud', 'bounding_box']
        }])

    def _get_subset(self, subset):
        if subset == DatasetSubset.TRAIN:
            return 'train'
        elif subset == DatasetSubset.VAL:
            return 'val'
        else:
            return 'test'

    def _get_file_list(self, cfg, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            logging.info('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']))
            samples = dc[subset]

            for s in tqdm(samples, leave=False):
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_cloud_path': cfg.DATASETS.KITTI.PARTIAL_POINTS_PATH % s,
                    'bounding_box_path': cfg.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH % s,
                })

        logging.info('Complete collecting files of the dataset. Total files: %d' % len(file_list))
        return file_list


# //////////////////////////////////////////// = Dataset Loader Mapping = //////////////////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'Completion3D': Completion3DDataLoader,
    'ShapeNet': ShapeNetDataLoader,
    'ShapeNetCars': ShapeNetCarsDataLoader,
    'KITTI': KittiDataLoader
}  # yapf: disable