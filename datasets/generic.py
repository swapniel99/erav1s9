import os
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from abc import abstractmethod


class DataSet(object):
    mean = None
    std = None
    classes = None

    def __init__(self, batch_size=32, augment_transforms=None, shuffle=True):
        self.batch_size = batch_size
        self.augment_transforms = augment_transforms
        self.shuffle = shuffle
        self.loader_kwargs = {'batch_size': batch_size, 'num_workers': os.cpu_count()}
        self.std_transforms = [
            ToTensor(),
            A.Normalize(self.mean, self.std)
        ]
        self.train_transforms = None
        self.test_transforms = None
        self.train_loader, self.test_loader = self.get_loaders()
        self.example_iter = iter(self.train_loader)

    def get_train_transforms(self):
        all_transforms = list()
        if self.augment_transforms is not None:
            all_transforms += self.augment_transforms
        all_transforms += self.std_transforms
        return lambda x: A.Compose(all_transforms)(image=np.array(x))['image']

    def get_test_transforms(self):
        return lambda x: A.Compose(self.std_transforms)(image=np.array(x))['image']

    @abstractmethod
    def get_train_loader(self):
        self.train_transforms = self.get_train_transforms()

    @abstractmethod
    def get_test_loader(self):
        self.test_transforms = self.get_test_transforms()

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()

    @classmethod
    def denormalise(cls, tensor):
        for t, m, s in zip(tensor, cls.mean, cls.std):
            t.mul_(s).add_(m)
        return tensor

    @abstractmethod
    def show_transform(self, img):
        pass

    def show_examples(self, figsize=None, denorm=True):
        batch_data, batch_label = next(self.example_iter)

        _ = plt.figure(figsize=figsize)
        for i in range(12):
            plt.subplot(3, 4, i + 1)
            plt.tight_layout()
            image = batch_data[i]
            if denorm:
                image = self.denormalise(image)
            plt.imshow(self.show_transform(image), cmap='gray')
            label = batch_label[i].item()
            if self.classes is not None:
                label = f'{label}:{self.classes[label]}'
            plt.title(str(label))
            plt.xticks([])
            plt.yticks([])
