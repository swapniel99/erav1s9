import torch
from torchvision import datasets
import albumentations as A

from .generic import DataSet


class MNIST(DataSet):
    mean = (0.1307,)
    std = (0.3081,)
    classes = None

    def get_train_transforms(self):
        self.augment_transforms = self.augment_transforms or [
            A.Rotate(limit=7, p=1.),
            A.Perspective(scale=0.3, p=0.5, fit_output=True)
        ]
        return super(MNIST, self).get_train_transforms()

    def get_train_loader(self):
        super(MNIST, self).get_train_loader()

        train_data = datasets.MNIST('../data', train=True, download=True, transform=self.train_transforms)
        self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=self.shuffle, **self.loader_kwargs)
        return self.train_loader

    def get_test_loader(self):
        super(MNIST, self).get_test_loader()

        test_data = datasets.MNIST('../data', train=False, download=True, transform=self.test_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **self.loader_kwargs)
        return self.test_loader

    def show_transform(self, img):
        return img.squeeze(0)
