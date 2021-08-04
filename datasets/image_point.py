import numpy as np
import open3d
import torch
from PIL import Image

from torchvision import transforms
from pathlib import Path

open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

import torch.utils.data as data


import torchvision.transforms as T


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INV_IMAGENET_MEAN = [-m for m in IMAGENET_MEAN]
INV_IMAGENET_STD = [1.0 / s for s in IMAGENET_STD]


def imagenet_preprocess():
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def rescale(x):
    lo, hi = x.min(), x.max()
    return x.sub(lo).div(hi - lo)


def imagenet_deprocess(rescale_image=True):
    transforms = [
        T.Normalize(mean=[0, 0, 0], std=INV_IMAGENET_STD),
        T.Normalize(mean=INV_IMAGENET_MEAN, std=[1.0, 1.0, 1.0]),
    ]
    if rescale_image:
        transforms.append(rescale)
    return T.Compose(transforms)


def image_to_numpy(img):
    return img.detach().cpu().mul(255).byte().numpy().transpose(1, 2, 0)


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n-pcd.shape[0])])
    return pcd[idx[:n]]


class RandomJitter(object):
    def __init__(self, sigma=0.02, clip=0.1):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(data.shape[0], 3), -1 * self.clip, self.clip)
        data[:, 0:3] += jitter
        return data


class ImageToPoint(data.Dataset):
    def __init__(self, d_path, split='train', im_size=128, points=4096):
        super().__init__()
        self.d_path = Path(d_path)
        assert(self.d_path.exists())
        assert(self.d_path.is_dir())

        self.split = split

        list_dir = self.d_path.joinpath('lists')
        points_dir = self.d_path.joinpath('points')
        im_dir = self.d_path.joinpath('renderings')

        assert(list_dir.exists())
        assert(points_dir.exists())
        assert(im_dir.exists())

        # classes
        classes_path = self.d_path.joinpath('classes.txt')

        self.class_to_id = {}
        self.id_to_class = {}

        with open(str(classes_path), 'r') as cls_file:
            for line in cls_file.readlines():
                self.class_to_id[line.split()[0]] = line.split()[1]
                self.id_to_class[line.split()[1]] = line.split()[0]
        #####

        self.im_size = im_size
        self.points = points

        self.data_pairs = []

        for category in list_dir.iterdir():
            if category.is_dir():
                split_list = category.joinpath(split + '.txt')
                assert(split_list.exists())
                # list of object ids
                with open(str(split_list), 'r') as split_file:
                    for object_id in split_file.readlines():
                        object_id = object_id.strip()
                        points_obj = points_dir.joinpath(category.name).joinpath(object_id)
                        im_obj = im_dir.joinpath(category.name).joinpath(object_id)

                        assert(points_obj.exists())
                        assert(im_obj.exists())

                        for img in im_obj.iterdir():
                            if img.suffix == '.png':
                                point = points_obj.joinpath(img.stem + '.ply')
                                assert(point.exists())
                                self.data_pairs.append((img, point))

    def __getitem__(self, index):
        image_path, pcd_path = self.data_pairs[index]

        pcd = np.asarray(open3d.io.read_point_cloud(str(pcd_path)).points)
        pcd = resample_pcd(pcd, self.points)

        image = pil_loader(str(image_path))

        if self.split == 'test':
            t = transforms.Compose([transforms.Resize(self.im_size),
                                    transforms.ToTensor(),
                                    imagenet_preprocess()])
        else:
            t = transforms.Compose([transforms.Resize(self.im_size),
                                    transforms.ToTensor(),
                                    imagenet_preprocess()])

        assert(pcd.shape[1] == 3)

        if self.split == 'test':
            return t(image), torch.from_numpy(pcd.astype(np.float32).T), pcd_path.parents[1].name
        else:
            return t(image), torch.from_numpy(pcd.astype(np.float32).T)

    def __len__(self):
        # return 16
        return len(self.data_pairs)
