import argparse
import os
import time
import numpy as np
import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils
import data
import models
import pandas as pd
import random
from collections import defaultdict
from datetime import datetime
from utils import eval_dataset

from robustbench.data import load_cifar10c, load_cifar10
from robustbench.utils import clean_accuracy


corruptions = ['shot_noise', 'motion_blur', 'snow', 'pixelate', 'gaussian_noise', 'defocus_blur',
                                 'brightness', 'fog', 'zoom_blur', 'frost', 'glass_blur', 'impulse_noise', 'contrast',
                                 'jpeg_compression', 'elastic_transform']


def corr_eval(x_corrs, y_corrs, model):
    model.eval()
    res = np.zeros((5, 15))
    for i in range(1, 6):
        for j, c in enumerate(corruptions):
            res[i-1, j] = clean_accuracy(model, x_corrs[i][j].cuda(), y_corrs[i][j].cuda())
            print(c, i, res[i-1, j])

    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'advprop', 'cnn', 'fc', 'linear', 'fastat'], type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--output', default='output.csv', type=str)
    parser.add_argument('--only-clean', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    x_clean, y_clean = load_cifar10(n_examples=10000, data_dir=args.data_dir)
    cifar_norm = True #False if args.augmix else True

    if args.model == 'resnet18':
        model = models.get_model(args.model, 10, False, data.shapes_dict['cifar10'], 64, 16,
                             1024, 'relu', cifar_norm=cifar_norm).cuda()
    
        if args.checkpoint != '':
            model.load_state_dict(torch.load(args.checkpoint)['last'])


    
    model.eval()

    clean_acc = clean_accuracy(model, x_clean.cuda(), y_clean.cuda())
    print("Clean accuracy: ", clean_acc)

    if args.only_clean:
        return 0

    x_corrs = []
    y_corrs = []
    x_corrs.append(x_clean)
    y_corrs.append(y_clean)
    for i in range(1, 6):
        x_corr = []
        y_corr = []
        for j, corr in enumerate(corruptions):
            x_, y_ = load_cifar10c(n_examples=10000, data_dir=args.data_dir, severity=i, corruptions=(corr,))
            x_corr.append(x_)
            y_corr.append(y_)
        x_corrs.append(x_corr)
        y_corrs.append(y_corr)
    x_corrs_fast = []
    y_corrs_fast = []
    for i in range(1, 6):
        x_, y_ = load_cifar10c(n_examples=1000, data_dir=args.data_dir, severity=i, shuffle=True)
        x_corrs_fast.append(x_)
        y_corrs_fast.append(y_)
    
    corr_res_last = corr_eval(x_corrs, y_corrs, model)
    corr_data_last = pd.DataFrame({i+1: corr_res_last[i, :] for i in range(0, 5)}, index=corruptions)
    corr_data_last.loc['average'] = {i+1: np.mean(corr_res_last, axis=1)[i] for i in range(0, 5)}
    corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
    corr_data_last.to_csv(args.output)
    print(corr_data_last)
    return 0

if __name__ == "__main__":
    main()