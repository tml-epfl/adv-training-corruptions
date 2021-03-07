import os
import argparse
import numpy as np
import torch
import models
import pandas as pd
import data

from robustbench.data import load_cifar10
from robustbench.utils import clean_accuracy


def corr_eval(x_corrs, y_corrs, model):
    model.eval()
    res = np.zeros((5, 15))
    for i in range(1, 6):
        for j, c in enumerate(data.corruptions):
            res[i-1, j] = clean_accuracy(model, x_corrs[i][j].cuda(), y_corrs[i][j].cuda())
            print(c, i, res[i-1, j])

    return res


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--output', default='output.csv', type=str)
    parser.add_argument('--only_clean', action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    x_clean, y_clean = load_cifar10(n_examples=10000, data_dir=args.data_dir)

    model = models.PreActResNet18(n_cls=10, model_width=64, cifar_norm=True).cuda()
    model.load_state_dict(torch.load(args.checkpoint)['last'])
    model.eval()

    clean_acc = clean_accuracy(model, x_clean.cuda(), y_clean.cuda())
    print("Clean accuracy: ", clean_acc)

    if args.only_clean:
        return

    x_corrs, y_corrs, _, _ = data.get_cifar10_numpy()
    
    corr_res_last = corr_eval(x_corrs, y_corrs, model)
    corr_data_last = pd.DataFrame({i+1: corr_res_last[i, :] for i in range(0, 5)}, index=data.corruptions)
    corr_data_last.loc['average'] = {i+1: np.mean(corr_res_last, axis=1)[i] for i in range(0, 5)}
    corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
    corr_data_last.to_csv(args.output)
    print(corr_data_last)


if __name__ == "__main__":
    main()
