
import torch
import csv
import argparse
import copy
from typing import List
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from torchvision import models as torchvision_models
from torch.hub import load_state_dict_from_url
from torch.tensor import Tensor
from torchvision.models import AlexNet
from robustness.datasets import DATASETS
from torchvision import datasets, transforms
import torch.nn as nn

from robustness.attacker import AttackerModel
from robustness.model_utils import make_and_restore_model

from data import ImageNet100C
import models
import pandas as pd
import numpy as np
import os

# Part of the code was taken from https://github.com/cassidylaidlaw/perceptual-advex


corruptions = ['shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
                                 'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
                                 'jpeg_compression', 'elastic_transform']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Common corruptions evaluation')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--output', type=str, default='default.csv',
                        help='output CSV')
    parser.add_argument('--only-clean', action='store_true')

    args = parser.parse_args()

    if args.output == 'default.csv':
        args.output = args.checkpoint.rsplit('.', 1)[0] + '_last.csv'

    if args.dataset == 'imagenet100c' and os.path.isfile(args.output):
        print(args.output, "is already evaluated")
        quit()

    dataset_cls = DATASETS[args.dataset]
    dataset = dataset_cls(
                args.dataset_path)

    model = models.get_model('resnet18', 1000, False, (10000,3,224,224), 64, 16, 1024,
                              'relu', cifar_norm=True).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['last'])


    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    

    if args.dataset == 'imagenet100':
        dataset = dataset_cls(
                args.dataset_path)
        _, val_loader = dataset.make_loaders(
                4, args.batch_size, only_val=True)
        if args.arch != 'resnet50':
            preprocess = transforms.Compose(
                    [transforms.ToTensor()])

            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess,
            ])
            val_dataset = val_loader.dataset
            val_dataset.transform = test_transform
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8, pin_memory=True)
        batches_correct: List[Tensor] = []

        for batch_index, (inputs, labels) in enumerate(val_loader):
            
            if (
                args.num_batches is not None and
                batch_index >= args.num_batches
            ):
                break
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                logits = model(inputs)
                batches_correct.append(
                    (logits.argmax(1) == labels).detach())

        accuracy = torch.cat(batches_correct).float().mean().item()
        print('Clean accuracy:', args.checkpoint, accuracy)
        quit()


    res = np.zeros((5, 15))

    for k, corruption_type in enumerate(corruptions):
        for severity in range(1, 6):
            print(f'CORRUPTION\t{corruption_type}\tseverity = {severity}')
            dataset = dataset_cls(
                args.dataset_path, corruption_type, severity)
            _, val_loader = dataset.make_loaders(
                4, args.batch_size, only_val=True)
            preprocess = transforms.Compose(
                [transforms.ToTensor()])
        
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                preprocess,
            ])
            val_dataset = val_loader.dataset
            val_dataset.transform = test_transform
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=8, pin_memory=True)
            batches_correct: List[Tensor] = []
            for batch_index, (inputs, labels) in enumerate(val_loader):
                if args.arch == 'resnet50':
                    inputs = normalizer(inputs)
                if (
                    args.num_batches is not None and
                    batch_index >= args.num_batches
                ):
                    break
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                with torch.no_grad():
                    logits = model(inputs)
                    batches_correct.append(
                        (logits.argmax(1) == labels).detach())

            accuracy = torch.cat(batches_correct).float().mean().item()

            print('OVERALL\t',
                f'accuracy = {accuracy * 100:.1f}',
                sep='\t')
            res[severity-1, k] = accuracy

    corr_data_last = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=corruptions)
    corr_data_last.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
    corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
    corr_data_last.to_csv(args.output)
    print(corr_data_last)
