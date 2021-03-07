import os
import argparse
import torch
from typing import List
from torch.tensor import Tensor
from torchvision import transforms
import models
import pandas as pd
import numpy as np
import data

# Part of the code was taken from https://github.com/cassidylaidlaw/perceptual-advex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Common corruptions evaluation')

    parser.add_argument('--dataset', type=str, default='imagenet100')
    parser.add_argument('--dataset_path', type=str, default='../imagenet')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--output', type=str, default='default.csv',
                        help='output CSV')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.output == 'default.csv':
        args.output = args.checkpoint.rsplit('.', 1)[0] + '_last.csv'

    dataset_cls = data.DATASETS[args.dataset]
    dataset = dataset_cls(args.dataset_path)

    model = models.PreActResNet18_I(n_cls=1000, model_width=64)
    model.load_state_dict(torch.load(args.checkpoint)['last'])
    model.cuda().eval()

    if args.dataset == 'imagenet100':
        dataset = dataset_cls(
                args.dataset_path)
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

    for k, corruption_type in enumerate(data.corruptions):
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

    corr_data_last = pd.DataFrame({i+1: res[i, :] for i in range(0, 5)}, index=data.corruptions)
    corr_data_last.loc['average'] = {i+1: np.mean(res, axis=1)[i] for i in range(0, 5)}
    corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
    corr_data_last.to_csv(args.output)
    print(corr_data_last)
