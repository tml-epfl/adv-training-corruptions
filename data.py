import torch.utils.data as td
from torchvision import datasets, transforms

import torch
import tempfile
import os
import numpy as np

from robustness.datasets import CIFAR, DATASETS, DataSet, CustomImageNet
from robustbench.data import load_cifar10c, load_cifar10

# Loaders for Imagenet100 were taken from https://github.com/cassidylaidlaw/perceptual-advex

class ImageNet100(CustomImageNet):
    def __init__(self, data_path, **kwargs):

        super().__init__(
            data_path=data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


class ImageNet100A(CustomImageNet):
    def __init__(self, data_path, **kwargs):
        super().__init__(
            data_path=data_path,
            custom_grouping=[
                [],
                [],
                [],
                [8],
                [],
                [13],
                [],
                [15],
                [],
                [20],
                [],
                [28],
                [],
                [32],
                [],
                [36],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [53],
                [],
                [64],
                [],
                [],
                [],
                [],
                [],
                [],
                [75],
                [],
                [83],
                [86],
                [],
                [],
                [],
                [94],
                [],
                [],
                [],
                [],
                [],
                [104],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [125],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [150],
                [],
                [],
                [],
                [159],
                [],
                [],
                [167],
                [],
                [170],
                [172],
                [174],
                [176],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [194],
                [],
            ],
            **kwargs,
        )


class ImageNet100C(CustomImageNet):
    """
    ImageNet-C, but restricted to the ImageNet-100 classes.
    """

    def __init__(
        self,
        data_path,
        corruption_type: str = 'gaussian_noise',
        severity: int = 1,
        **kwargs,
    ):
        # Need to create a temporary directory to act as the dataset because
        # the robustness library expects a particular directory structure.
        tmp_data_path = tempfile.mkdtemp()
        os.symlink(os.path.join(data_path, corruption_type, str(severity)),
                   os.path.join(tmp_data_path, 'test'))

        super().__init__(
            data_path=tmp_data_path,
            custom_grouping=[[label] for label in range(0, 1000, 10)],
            **kwargs,
        )


DATASETS['imagenet100'] = ImageNet100
DATASETS['imagenet100a'] = ImageNet100A
DATASETS['imagenet100c'] = ImageNet100C


def get_loaders(dataset, n_ex, batch_size, train_set, shuffle, data_augm, n_train=-1, p_label_noise=0.0, drop_last=False):
    dir_ = '../data/'
    dataset_f = datasets_dict[dataset]
    batch_size = n_ex if n_ex < batch_size and n_ex != -1 else batch_size
    num_workers = 15

    if dataset == 'imagenet100':
        
        if dataset == 'imagenet100':
            imset = dataset_f(dir_ + '/imagenet/')

            
        if n_ex != -1:
            train_loader, val_loader = imset.make_loaders(num_workers, batch_size, subset=n_ex)
        else:
            train_loader, val_loader = imset.make_loaders(num_workers, batch_size)
        
        
        preprocess = transforms.Compose(
            [transforms.ToTensor()])
        
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             preprocess])
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            preprocess,
        ])
        train_dataset = train_loader.dataset
        train_dataset.transform = train_transform
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, pin_memory=True)
        val_dataset = val_loader.dataset
        val_dataset.transform = test_transform
        #print(val_dataset.transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers, pin_memory=True)

        if train_set:
            return train_loader
        else:
            return val_loader

    data_augm_transforms = [transforms.RandomCrop(32, padding=4)]
    if dataset not in ['mnist', 'svhn']:
        #data_augm_transforms.append(transforms.RandomHorizontalFlip())
        data_augm_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(.25,.25,.25),
            #transforms.RandomRotation(2)
        ]
    transform_list = data_augm_transforms if data_augm else []
    transform = transforms.Compose(transform_list + [transforms.ToTensor()])

    if 'binary' in dataset:
        cl1, cl2 = 1, 7  # cifar10: 1=auto, 7=horse
    if train_set:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=True, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='train', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex
        n_cls = max(data.targets) + 1

        if 'binary' in dataset:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            # data.targets = list(data.targets)
            n_ex = len(data.targets) if n_ex == -1 else n_ex
            n_cls = 2
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        if dataset == 'svhn':
            data.targets = data.labels
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        if n_train > 0:
            indices = np.random.permutation(np.arange(n_ex))[:n_train]
            data.data, data.targets = data.data[indices], data.targets[indices]
            n_ex = n_train

        data.label_noise = np.zeros(n_ex, dtype=bool)
        if p_label_noise > 0.0:
            # gen random indices
            indices = np.random.permutation(np.arange(n_ex))[:int(n_ex*p_label_noise)]
            for index in indices:
                lst_classes = list(range(n_cls))
                lst_classes.remove(data.targets[index].item())
                data.targets[index] = np.random.choice(lst_classes)
            data.label_noise[indices] = True

        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
                                             num_workers=num_workers, drop_last=drop_last)
    else:
        if dataset != 'svhn':
            data = dataset_f(dir_, train=False, transform=transform, download=True)
        else:
            data = dataset_f(dir_, split='test', transform=transform, download=True)
        n_ex = len(data) if n_ex == -1 else n_ex

        if 'binary' in dataset:
            data.targets = np.array(data.targets)
            idx = (data.targets == cl1) + (data.targets == cl2)
            data.data, data.targets = data.data[idx], data.targets[idx]
            data.targets[data.targets == cl1], data.targets[data.targets == cl2] = 0, 1
            data.targets = list(data.targets)  # to reduce memory consumption
        if '_gs' in dataset:
            data.data = data.data.mean(3).astype(np.uint8)
        if dataset == 'svhn':
            data.targets = data.labels
        data.data, data.targets = data.data[:n_ex], data.targets[:n_ex]

        data.label_noise = np.zeros(n_ex)
        loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                                             num_workers=2, drop_last=drop_last)
    return loader


def create_loader(x, y, ln, n_ex, batch_size, shuffle, drop_last):
    if n_ex > 0:
        x, y, ln = x[:n_ex], y[:n_ex], ln[:n_ex]
    data = td.TensorDataset(x, y, ln)
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle, pin_memory=False,
                                         num_workers=2, drop_last=drop_last)
    return loader


def get_xy_from_loader(loader, cuda=True):
    tuples = [(x, y, ln) for (x, y, ln) in loader]
    x_vals = torch.cat([x for (x, y, ln) in tuples])
    y_vals = torch.cat([y for (x, y, ln) in tuples])
    ln_vals = torch.cat([ln for (x, y, ln) in tuples])
    if cuda:
        x_vals, y_vals, ln_vals = x_vals.cuda(), y_vals.cuda(), ln_vals.cuda()
    return x_vals, y_vals, ln_vals


def get_cifar10_numpy():
    x_clean, y_clean = load_cifar10(n_examples=10000, data_dir='../data')
    x_corrs = []
    y_corrs = []
    x_corrs.append(x_clean)
    y_corrs.append(y_clean)
    for i in range(1, 6):
        x_corr = []
        y_corr = []
        for j, corr in enumerate(corruptions):
            x_, y_ = load_cifar10c(n_examples=10000, data_dir='../data', severity=i, corruptions=(corr,))
            x_corr.append(x_)
            y_corr.append(y_)

        x_corrs.append(x_corr)
        y_corrs.append(y_corr)

    x_corrs_fast = []
    y_corrs_fast = []
    for i in range(1, 6):
        x_, y_ = load_cifar10c(n_examples=1000, data_dir='../data', severity=i, shuffle=True)
        x_corrs_fast.append(x_)
        y_corrs_fast.append(y_)

    return x_corrs, y_corrs, x_corrs_fast, y_corrs_fast


datasets_dict = {'mnist': datasets.MNIST, 'mnist_binary': datasets.MNIST, 'svhn': datasets.SVHN, 'cifar10': datasets.CIFAR10,
                 'cifar10_binary': datasets.CIFAR10, 'cifar10_binary_gs': datasets.CIFAR10, 'imagenet100': ImageNet100,
                 'SIN': ImageNet100
                 }
shapes_dict = {'mnist': (60000, 1, 28, 28), 'mnist_binary': (13007, 1, 28, 28), 'svhn': (73257, 3, 32, 32),
               'cifar10': (50000, 3, 32, 32), 'cifar10_binary': (10000, 3, 32, 32),
               'cifar10_binary_gs': (10000, 1, 32, 32), 'uniform_noise': (1000, 1, 28, 28), 'imagenet100': (10000,3,224,224)
               }
classes_dict = {'cifar10': {0: 'airplane',
                            1: 'automobile',
                            2: 'bird',
                            3: 'cat',
                            4: 'deer',
                            5: 'dog',
                            6: 'frog',
                            7: 'horse',
                            8: 'ship',
                            9: 'truck',
                            }
                }
corruptions = ['shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
               'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
               'jpeg_compression', 'elastic_transform']
