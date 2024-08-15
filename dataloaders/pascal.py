"""
Load pascal VOC dataset
"""
import io
import os

import imageio
import numpy as np
from PIL import Image
import torch

from util.utils import myslic
from dataloaders.common import BaseDataset
import cv2 as cv


class VOC(BaseDataset):
    """
    Base Class for VOC Dataset

    Args:
        base_dir:
            VOC dataset directory
        split:
            which split to use
            choose from ('train', 'val', 'trainval', 'trainaug')
        transform:
            transformations to be performed on images/masks
        to_tensor:
            transformation to convert PIL Image to tensor
    """

    def __init__(self, base_dir, split, transforms=None, to_tensor=None, slic=None):
        super().__init__(base_dir)

        self.split = split
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')  # 原始图片17125张
        self._label_dir = os.path.join(self._base_dir, 'SegmentationClassAug')  # 2913张，语义mask，语义分类
        self._inst_dir = os.path.join(self._base_dir, 'SegmentationObjectAug')  # 实体分割 10582张
        self._scribble_dir = os.path.join(self._base_dir, 'ScribbleAugAuto')  # 若监督绘画 12031张，这个目前是没有用的
        self._id_dir = os.path.join(self._base_dir, 'ImageSets',
                                    'Segmentation')  # trainaug.txt提供相同类的图片，总共20类（20个txt），同一类在同一个txt
        # @GL slic
        self.slic = slic
        self.transforms = transforms
        self.to_tensor = to_tensor

        with open(os.path.join(self._id_dir, f'{self.split}.txt'), 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        # Fetch data
        id_ = self.ids[idx]
        path = os.path.join(self._image_dir, f'{id_}.jpg')
        image = Image.open(path)
        # @GL
        """
        不做判断替换了，直接将slic结果加入sample，同时读进去4张，在model时选择(slic的输入维度是否和此处image一致？)
        """
        img_arr = myslic(cv.imread(path))

        slic_image = Image.fromarray(np.uint8(img_arr))
        if slic_image.mode == 'L':
            slic_image = slic_image.convert('RGB')
        semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}.png'))
        instance_mask = Image.open(os.path.join(self._inst_dir, f'{id_}.png'))
        scribble_mask = Image.open(os.path.join(self._scribble_dir, f'{id_}.png'))
        sample = {'image': image,  # or slic_image
                  'img_slic': slic_image,
                  'label': semantic_mask,
                  'inst': instance_mask,
                  'scribble': scribble_mask}

        # Image-level transformation
        if self.transforms is not None:
            sample = self.transforms(sample)
        # Save the original image (without normalization)
        image_t = torch.from_numpy(
            np.array(sample['image']).transpose(2, 0, 1))  # plt.imshow()中则是[H, W, Channels]pytorch中是[CHW] [3, 448, 448]
        x = sample['img_slic']
        slic_image_t = torch.from_numpy(
            np.array(sample['img_slic']).transpose(2, 0, 1))  # plt.imshow()中则是[H, W, Channels]pytorch中是[CHW]
        # Transform to tensor
        if self.to_tensor is not None:
            sample = self.to_tensor(sample)

        sample['id'] = id_
        sample['image_t'] = image_t
        sample['img_slic_t'] = slic_image_t
        # Add auxiliary attributes
        for key_prefix in self.aux_attrib:
            # Process the data sample, create new attributes and save them in a dictionary
            aux_attrib_val = self.aux_attrib[key_prefix](sample, **self.aux_attrib_args[key_prefix])
            for key_suffix in aux_attrib_val:
                # one function may create multiple attributes, so we need suffix to distinguish them
                sample[key_prefix + '_' + key_suffix] = aux_attrib_val[key_suffix]

        return sample
