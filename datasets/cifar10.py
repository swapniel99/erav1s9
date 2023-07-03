import numpy as np
import cv2
import torch
from torchvision import datasets
import albumentations as A

from .generic import MyDataSet


class AlbCIFAR10(datasets.CIFAR10):
    def __init__(self, root, alb_transform=None, **kwargs):
        super(AlbCIFAR10, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        image, label = super(AlbCIFAR10, self).__getitem__(index)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return image, label


class CIFAR10(MyDataSet):
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None

    def get_train_transforms(self):
        if self.alb_transforms is None:
            self.alb_transforms = [
                A.ToGray(p=0.2),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                # Padding value doesnt matter here.
                A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
                # Since normalisation was the first step, mean is already 0, so cutout fill_value = 0
                A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=0.5),
                A.CenterCrop(32, 32, p=1)
            ]
        return super(CIFAR10, self).get_train_transforms()

    def get_train_loader(self):
        super(CIFAR10, self).get_train_loader()

        train_data = AlbCIFAR10('../data', train=True, download=True, alb_transform=self.train_transforms)
        if self.classes is None:
            self.classes = {i: c for i, c in enumerate(train_data.classes)}
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=self.shuffle, **self.loader_kwargs)
        return self.train_loader

    def get_test_loader(self):
        super(CIFAR10, self).get_test_loader()

        test_data = AlbCIFAR10('../data', train=False, download=True, alb_transform=self.test_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **self.loader_kwargs)
        return self.test_loader

    def show_transform(self, img):
        return img.permute(1, 2, 0)
