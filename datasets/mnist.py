import torch
from torchvision import datasets, transforms

from .generic import DataSet


class MNIST(DataSet):
    mean = (0.1307,)
    std = (0.3081,)
    classes = None

    def get_train_loader(self):
        self.augment_transforms = self.augment_transforms or transforms.Compose([
            transforms.RandomRotation(7),
            transforms.RandomPerspective(0.3, 0.5)
        ])
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
