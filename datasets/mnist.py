import numpy as np
import torch
from torchvision import datasets
import albumentations as A

from .generic import MyDataSet


class AlbMNIST(datasets.MNIST):
    def __init__(self, root, alb_transform=None, **kwargs):
        super(AlbMNIST, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        image, label = super(AlbMNIST, self).__getitem__(index)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return image, label


class MNIST(MyDataSet):
    mean = (0.1307,)
    std = (0.3081,)
    classes = None

    def get_train_transforms(self):
        if self.alb_transforms is None:
            self.alb_transforms = [
                A.Rotate(limit=7, p=1.),
                A.Perspective(scale=0.2, p=0.5, fit_output=False)
            ]
        return super(MNIST, self).get_train_transforms()

    def get_train_loader(self):
        super(MNIST, self).get_train_loader()

        train_data = AlbMNIST('../data', train=True, download=True, alb_transform=self.train_transforms)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=self.shuffle, **self.loader_kwargs)
        return self.train_loader

    def get_test_loader(self):
        super(MNIST, self).get_test_loader()

        test_data = AlbMNIST('../data', train=False, download=True, alb_transform=self.test_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **self.loader_kwargs)
        return self.test_loader

    def show_transform(self, img):
        return img.squeeze(0)
