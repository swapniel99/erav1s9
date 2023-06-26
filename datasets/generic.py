import os
from abc import abstractmethod
from matplotlib import pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MyDataSet(object):
    mean = None
    std = None
    classes = None

    def __init__(self, batch_size=32, alb_transforms=None, shuffle=True):
        self.batch_size = batch_size
        self.alb_transforms = alb_transforms
        self.shuffle = shuffle
        self.loader_kwargs = {'batch_size': batch_size, 'num_workers': os.cpu_count(), 'pin_memory': True}
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()
        self.example_iter = iter(self.train_loader)

    def get_train_transforms(self):
        all_transforms = [A.Normalize(self.mean, self.std)]
        if self.alb_transforms is not None:
            all_transforms += self.alb_transforms
        all_transforms.append(ToTensorV2())
        return A.Compose(all_transforms)

    def get_test_transforms(self):
        all_transforms = [A.Normalize(self.mean, self.std), ToTensorV2()]
        return A.Compose(all_transforms)

    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_test_loader(self):
        pass

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
