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
from utils import rob_err, clamp, eval_dataset
import logging

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

    return res

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--data_dir', default='../cifar-data', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'mnist_binary', 'svhn', 'cifar10', 'cifar10_binary', 'cifar10_binary_gs',
                                                                 'uniform_noise', 'imagenet100'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet50','resnet18', 'lenet', 'cnn', 'fc', 'linear'], type=str)
    parser.add_argument('--epochs', default=30, type=int,
                        help='15 epochs to reach 45% adv err, 30 epochs to reach the reported clean/adv errs')
    parser.add_argument('--lr_schedule', default='piecewise', choices=['cyclic', 'piecewise'])
    parser.add_argument('--lr_max', default=0.1, type=float, help='0.05 in Table 1, 0.2 in Figure 2')
    parser.add_argument('--attack', default='none', type=str, choices=['pgd', 'pgd_corner', 'fgsm', 'random_noise', 'free', 'none', 'sigma', 'rlat', 'random_gs'])
    parser.add_argument('--eps', default=8.0, type=float)
    parser.add_argument('--attack_iters', default=1, type=int, help='n_iter of pgd for evaluation')
    parser.add_argument('--pgd_train_n_iters', default=1, type=int, help='n_iter of pgd for training (if attack=pgd)')
    parser.add_argument('--pgd_alpha_train', default=0.5, type=float)
    parser.add_argument('--normreg', default=0.0, type=float) # Can be used for numerical stability
    parser.add_argument('--grad-reg', default=0.00, type=float)
    parser.add_argument('--distance', default='linf', type=str)
    parser.add_argument('--model_path', default='models/default.pt', type=str)
    parser.add_argument('--eps-policy', default='scaling', type=str)
    parser.add_argument('--grow-policy', default='descending', type=str)
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--layers-policy', default='all', type=str)
    parser.add_argument('--layers', default='default', type=str)
    parser.add_argument('--without-input', action='store_true')
    parser.add_argument('--n_train', default=-1, type=int, help='Number of training points.')
    parser.add_argument('--n_rm_pts', default=0, type=float, help='Fraction of points to remove from the training set.')
    parser.add_argument('--fgsm_alpha', default=1.0, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay aka l2 regularization')
    parser.add_argument('--epoch_rm_noise', default=-1, type=float, help='remove noise at epoch epoch_rm_noise*epochs')
    parser.add_argument('--attack_init', default='zero', choices=['zero', 'random'])
    parser.add_argument('--random_grad_reg', default='random_uniform', choices=['random_uniform', 'random_noise'], help='at which point to take the 2nd grad (and also the 1st if enabled)')
    parser.add_argument('--activation', default='relu', type=str, help='currently supported only for resnet. relu or softplusA where A corresponds to the softplus alpha')
    parser.add_argument('--optimizer', default='sgdm', choices=['sgd', 'sgdm', 'adam'], type=str, help='Optimizer.')
    parser.add_argument('--rm_criterion', default='loss', choices=['loss', 'entropy'], type=str, help='Points with the highest values of this metric will be removed from the train set.')
    parser.add_argument('--loss', default='cross_entropy', choices=['cross_entropy', 'squared'], type=str, help='Loss.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--adv-part', default=1.0, type=float)
    parser.add_argument('--batch-part', default=1.0, type=float)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--export_images', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--final_adv_eval', action='store_true')
    parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision [not recommended]')
    parser.add_argument('--no_data_augm', action='store_true')
    parser.add_argument('--eval_early_stopped_model', action='store_true', help='whether to evaluate the model obtained via early stopping')
    parser.add_argument('--eval_iter_freq', default=1000, type=int, help='how often to evaluate test stats')
    parser.add_argument('--n_eval_every_k_iter', default=1024, type=int, help='on how many examples to eval every k iters')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers (for model in [fc, cnn])')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#hidden units on each conv layer (for model==fc)')
    parser.add_argument('--batch_size_eval', default=128, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--n_final_eval', default=1000, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')
    return parser.parse_args()

def delta_forward(model, X, deltas, layers):
    
    i = 0
    def out_hook(m, inp, out_layer):
        nonlocal i
        if layers[i] == model.normalize:
            new_out = (torch.clamp(inp[0] + deltas[i], 0, 1) - model.mu) / model.std
        else:
            new_out = out_layer + deltas[i]
        i += 1
        return new_out
    
    handles = [layer.register_forward_hook(out_hook) for layer in layers]
    out = model(X)

    for handle in handles:
        handle.remove()
    return out

def get_output_of_layers(model, X, layers):
    outputs = []
    def out_hook(m, inp, out):
        outputs.append(out)
    
    handles = [layer.register_forward_hook(out_hook) for layer in layers]
    out = model(X)

    for handle in handles:
        handle.remove()
    return outputs

def shape_hook(m, inp, out):
    delta_shapes.append(out.shape)

def main():
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cur_timestamp = str(datetime.now())[:-3]  # include also ms to prevent the probability of name collision
    model_width = {'linear': '', 'fc': args.n_hidden_fc, 'cnn': args.n_filters_cnn, 'lenet': '', 'resnet18': '', 'resnet50': ''}[args.model]
    model_str = '{}{}'.format(args.model, model_width)
    model_name = '{} dataset={} model={} eps={} attack={} epochs={} batch_size={} lr_max={} model_width={} weight_decay={} n_train={} epoch_rm_noise={} n_rm_pts={} seed={}'.format(
        cur_timestamp, args.dataset, model_str, args.eps, args.attack, args.epochs, args.batch_size, args.lr_max,
        args.model_width, args.weight_decay, args.n_train, args.epoch_rm_noise, args.n_rm_pts, args.seed)
    logger = utils.configure_logger(model_name, args.debug)
    logger.info(args)
    half_prec = args.half_prec
    n_cls = 2 if 'binary' in args.dataset else 10
    if args.dataset == 'imagenet100':
        n_cls = 1000

    if not os.path.exists('exps'): os.makedirs('exps')
    if not os.path.exists('images'): os.makedirs('images')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True

    if args.activation == 'softplus':  # only implemented for resnet18 currently
        assert args.model == 'resnet18'

    args.pgd_alpha = args.eps / 4
    eps, pgd_alpha, pgd_alpha_train = args.eps, args.pgd_alpha, args.pgd_alpha_train

    train_data_augm = False if args.dataset in ['mnist', 'mnist_binary'] else True
    train_batches = data.get_loaders(args.dataset, -1, args.batch_size, train_set=True, shuffle=True, data_augm=train_data_augm, n_train=args.n_train, drop_last=True)
    train_batches_fast = data.get_loaders(args.dataset, args.n_eval_every_k_iter, args.batch_size_eval, train_set=True, shuffle=False, data_augm=False, n_train=args.n_train, drop_last=False)
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False, drop_last=False)
    test_batches_fast = data.get_loaders(args.dataset, args.n_eval_every_k_iter, args.batch_size_eval, train_set=False, shuffle=False, data_augm=False, drop_last=False)

    cifar_norm = True
    if args.dataset == 'cifar10':
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

    model = models.get_model(args.model, n_cls, half_prec, data.shapes_dict[args.dataset], args.model_width, args.n_filters_cnn,
                             args.n_hidden_fc, args.activation, cifar_norm=cifar_norm).cuda()

    if args.parallel:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.attack == 'rlat':
        if args.model == 'resnet18' or args.model == 'resnet50':
            layers_dict = models.get_layers_dict(model)
        delta_shapes = []
        layers = layers_dict[args.layers]
        delta_max = len(layers)

        handles = [layer.register_forward_hook(shape_hook) for layer in layers]
        if args.dataset == 'cifar10':
            out = model(torch.zeros((args.batch_size, 3, 32, 32)).cuda())
        else:
            out = model(torch.zeros((args.batch_size, 3, 224, 224)).cuda())

        for handle in handles:
            handle.remove()


        if args.eps_policy == 'scaling':
            if args.dataset == 'cifar10':
                eps_start = eps / 3072
            else:
                eps_start = eps / (224 * 224 * 3)
            epses = [eps_start * (shp.numel() / shp[0]) for shp in delta_shapes]

        if args.eps_policy == 'lpips':
            eps_start = eps * 32 * 32
            epses = [eps_start / (shp.numel() / (shp[0] * shp[1])) for shp in delta_shapes]
        elif args.eps_policy == 'constant' or args.eps_policy == 'shared':
            epses = [eps for shp in delta_shapes]
        elif args.eps_policy == 'adaptive':
            eps_scale_factor = eps / (0.5 * np.sqrt(3072))

        if args.grow_policy == 'ascending':
            epses = [eps * (i + 1) for i, eps in enumerate(epses)]
        elif args.grow_policy == 'descending':
            epses = [eps / (i + 1) for i, eps in enumerate(epses)]
        elif args.grow_policy == 'lindescending':
            epses = [eps * ((delta_max - i) / delta_max) for i, eps in enumerate(epses)]

        print('Eps: ', epses)
    dist = args.distance

    if args.checkpoint != '':
        model.load_state_dict(torch.load(args.checkpoint)['last'])

    model.train()


    params = model.parameters()

    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.0, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgdm':
        opt = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(params, lr=args.lr_max, weight_decay=args.weight_decay)
    else:
        raise ValueError('decide about the right optimizer for the new model')

    if half_prec:
        model, opt = amp.initialize(model, opt, opt_level="O1")

    lr_schedule = utils.get_lr_schedule(args.lr_schedule, args.epochs, args.lr_max)
    loss_function = nn.CrossEntropyLoss() if args.loss == 'cross_entropy' else nn.MSELoss()

    metr_dict = defaultdict(list, vars(args))
    test_err_best, best_state_dict = 1.0, copy.deepcopy(model.state_dict())
    start_time = time.time()
    time_train, iteration, best_iteration = 0, 0, 0
    train_loss, train_err_clean, train_n_clean, grad_norm_x, avg_delta_l2 = \
        0, 0, 0, 0, 0
    margin_clean, margin_ln = 0, 0
    for epoch in range(args.epochs + 1):
        grad_norm_avg = 0        


        for i, (X, y) in enumerate(train_batches):
            
            X, y = X.cuda(), y.cuda()
            

            adv_step = True

            if eps == 0.0:
                adv_step = False


            if epoch == 0 and i > 0:  # epoch=0 runs only for one iteration (to check the training stats at init)
                break

            time_start_iter = time.time()
            lr = lr_schedule(epoch - 1 + (i + 1) / len(train_batches))  # epoch - 1 since the 0th epoch is skipped
            opt.param_groups[0].update(lr=lr)
            if adv_step:
                if args.attack == 'pgd':
                    pgd_rs = True if args.attack_init == 'random' else False
                    n_eps_warmup_epochs = 5
                    n_iterations_max_eps = n_eps_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                    eps_pgd_train = min(iteration / n_iterations_max_eps * eps, eps) if args.dataset == 'svhn' else eps
                    delta = utils.attack_pgd_training(
                        model, X, y, eps_pgd_train, pgd_alpha_train, opt, half_prec, args.pgd_train_n_iters, rs=pgd_rs, dist=dist)

                elif args.attack == 'fgsm':
                    if args.attack_init == 'zero':
                        delta = torch.zeros_like(X, requires_grad=True)
                    elif args.attack_init == 'random':
                        delta = utils.get_uniform_delta(X.shape, eps, requires_grad=True)
                    else:
                        raise ValueError('wrong args.attack_init')

                    X_adv = clamp(X + delta, 0, 1)
                    output = model(X_adv)
                    loss = F.cross_entropy(output, y)
                    grad = torch.autograd.grad(loss, delta)[0].detach()
                    argmax_delta = eps * utils.sign(grad)

                    n_alpha_warmup_epochs = 5
                    n_iterations_max_alpha = n_alpha_warmup_epochs * data.shapes_dict[args.dataset][0] // args.batch_size
                    fgsm_alpha = min(iteration / n_iterations_max_alpha * args.fgsm_alpha,
                                     args.fgsm_alpha) if args.dataset == 'svhn' else args.fgsm_alpha
                    delta.data = clamp(delta.data + fgsm_alpha * argmax_delta, -eps, eps)
                    delta.data = clamp(X + delta.data, 0, 1) - X
                    delta = delta.detach()

                elif args.attack == 'random_noise':
                    delta = utils.get_uniform_delta(X.shape, eps, requires_grad=False, dist=dist)
                
                elif args.attack == 'random_gs':
                    delta = torch.randn(X.size()).cuda() * eps
                    delta.data = clamp(X + delta.data, 0, 1) - X

                elif args.attack == 'none':
                    if (args.grad_reg != 0):
                        delta = torch.zeros_like(X, requires_grad=True)
                    else:
                        delta = torch.zeros_like(X, requires_grad=False)

                elif args.attack == 'sigma':
                    delta, out_adv = sigma_adv_attack(X, y, model, -1, 0.003, iter_n=40)
                    delta = delta.detach()

                elif args.attack == 'rlat':

                    deltas = [torch.zeros(delta_shapes[i], requires_grad=True, device=torch.device('cuda')) for i in range(delta_max)]

                    if args.eps_policy == "adaptive":

                        avg_norms = [3072 * 0.5]
                        def norm_hook(m, inp, out):
                            out_norms = (out**2).sum(tuple(range(1, out.ndim)))**0.5
                            avg_norms.append(torch.mean(out_norms))

                        handles = [layer.register_forward_hook(norm_hook) for layer in layers[1:]]

                        output = model(X, delta=deltas)

                        for handle in handles:
                            handle.remove()

                        epses = [eps_scale_factor * avg_norm for avg_norm in avg_norms]

                    else:
                        if args.layers != 'default':
                            output = delta_forward(model, X, deltas, layers)
                        else:
                            for param in model.parameters():
                                param.requires_grad = False
                            output = model(X, delta=deltas)

                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    if args.layers_policy == 'all':
                        grads = [deltas[_].grad.detach() for _ in range(delta_max)]
                        ri = -1
                    elif args.layers_policy == 'random':
                        ri = np.random.randint(delta_max)
                        grads[ri] = torch.autograd.grad(loss, deltas[ri])[0].detach()

                    for param in model.parameters(): # 10% faster, no need to store grad
                        param.requires_grad = True

                    if args.grad_reg != 0.0:
                        grad_norm_avg += grads[0].view(args.batch_size, -1).norm(2, 1).mean()

                    if dist == 'linf':
                        update_grads = [epses[i] * utils.sign(grad) for i, grad in enumerate(grads)]
                    elif dist == 'l2':
                        grad_norms = [(grad**2).sum(tuple(range(1, grad.ndim)), keepdim=True)**0.5 for grad in grads]
                        if args.eps_policy == 'shared':
                            grad_norm = torch.sqrt(sum([g**2 for g in grad_norms]))
                            update_grads = [epses[0] * grad / (grad_norm + args.normreg) for i, grad in enumerate(grads)]
                        else:
                            if args.layers_policy == 'random':
                                update_grads = [epses[i] * grad / (grad_norms[i] + args.normreg) if i == ri else grad for i, grad in enumerate(grads)]
                            else:
                                update_grads = [epses[i] * grad / (grad_norms[i] + args.normreg) for i, grad in enumerate(grads)]

                    for i, delta in enumerate(deltas):
                        if dist == 'linf':
                            deltas[i].data = clamp(delta.data + update_grads[i], -eps, eps)
                        if dist == 'l2':
                            deltas[i].data = delta.data + update_grads[i]
                        deltas[i] = deltas[i].detach()
                        batch_clean = int((1 - args.batch_part) * deltas[0].shape[0])
                        deltas[i][0:batch_clean] = 0
                        deltas[i].requires_grad = False
                else:
                    raise ValueError('wrong args.attack')
                
                if args.attack == 'rlat':
                    if (args.without_input):
                        deltas[0] = 0
                    else:
                        if args.layers != 'default':
                            output = delta_forward(model, X, deltas, layers)
                        else:
                            output = model(X, delta=deltas, ri=ri)
                else:
                    if args.batch_part != 1.0:
                        batch_clean = int((1 - args.batch_part) * delta.shape[0])
                        delta[0:batch_clean] = 0
                    output = model(X + delta)
            else:
                output = model(X)

            loss = loss_function(output, y)
            #print(loss)
            if args.grad_reg != 0.0:
                g = torch.autograd.grad(loss, delta, create_graph=True, retain_graph=True)[0] * args.batch_size
                g = g.view(args.batch_size, -1)
                grad_norm = g.norm(2, 1).mean()
                #print(grad_penalty, loss)
                loss = grad_norm * args.grad_reg + loss
                grad_norm_avg += grad_norm.detach()


            if epoch != 0:  # epoch == 0: only evaluation
                opt.zero_grad()
                utils.backward(loss, opt, half_prec)
                opt.step()
            else:
                opt.zero_grad()
                utils.backward(loss, opt, half_prec)


            time_train += time.time() - time_start_iter
            train_loss += loss.item() * y.size(0)
            train_err_clean += (output.max(1)[1] != y).sum().item()
            train_n_clean += y.size(0)
            if iteration % args.eval_iter_freq == 0:
                train_loss = train_loss / train_n_clean
                grad_norm_report = grad_norm_avg * args.batch_size / train_n_clean
                train_err_clean = train_err_clean / train_n_clean if train_n_clean > 0 else 0

                # it'd be incorrect to recalculate the BN stats on the test sets and for clean / adversarial points
                model.eval()
                test_err, _ = eval_dataset(test_batches_fast, model)

                time_elapsed = time.time() - start_time

                if test_err < test_err_best:  # save the best model
                    best_state_dict = copy.deepcopy(model.state_dict())
                    test_err_best, best_iteration = test_err, iteration
                
                train_str = '[train] loss {:.3f}, err_clean {:.2%}'.format(
                    train_loss, train_err_clean)
                test_str = '[test] err {:.2%}'.format(test_err)
                grad_str = '[grad] avg {:.2}'.format(grad_norm_report)
                best_str = '[best] err {:.2%}'.format(test_err_best)

                logger.info('{}-{}: {}  {} {} {} ({:.2f}m, {:.2f}m)'.format(epoch, iteration, train_str, test_str, grad_str,
                                                                      best_str, time_train/60, time_elapsed/60))

                metr_vals = [epoch, iteration, train_loss, train_err_clean,
                             test_err, time_train,
                             time_elapsed]
                metr_names = ['epoch', 'iter', 'train_loss', 'train_err_clean',
                              'test_err',
                              'time_train', 'time_elapsed']
                utils.update_metrics(metr_dict, metr_vals, metr_names)

                if not args.debug:
                    np.save('exps/{}.npy'.format(model_name), metr_dict)

                

                model.train()
                train_loss, train_err_clean, train_n_clean = 0, 0, 0

            iteration += 1

        freq_expensive_stats = 10
        if (epoch % freq_expensive_stats == 0 and epoch > 0) or (epoch == args.epochs):
            if epoch == args.epochs:  # only save and eval cifar10-c at the end
                model_str = args.model_path.rsplit('.', 1)[0]
                torch.save({'last': model.state_dict(), 'best': best_state_dict}, args.model_path)
                
                if args.dataset == 'cifar10':
                    model.load_state_dict(torch.load(args.model_path)['last'])
                    corr_res_last = corr_eval(x_corrs, y_corrs, model)
                    print('Last model on CIFAR10-C: {:.3%}'.format(np.mean(corr_res_last)))
                    corr_data_last = pd.DataFrame({i+1: corr_res_last[i, :] for i in range(0, 5)}, index=corruptions)
                    corr_data_last.loc['average'] = {i+1: np.mean(corr_res_last, axis=1)[i] for i in range(0, 5)}
                    corr_data_last['avg'] = corr_data_last[list(range(1,6))].mean(axis=1)
                    corr_data_last.to_csv(model_str + '_last.csv')
                    print(corr_data_last)

                    model.load_state_dict(torch.load(args.model_path)['best'])
                    corr_res_best = corr_eval(x_corrs, y_corrs, model)
                    print('Best model on CIFAR10-C: {:.3%}'.format(np.mean(corr_res_best)))
                    corr_data_best = pd.DataFrame({i+1: corr_res_best[i, :] for i in range(0, 5)}, index=corruptions)
                    corr_data_best.loc['average'] = {i+1: np.mean(corr_res_best, axis=1)[i] for i in range(0, 5)}
                    corr_data_best['avg'] = corr_data_best[list(range(1,6))].mean(axis=1)
                    corr_data_best.to_csv(model_str + '.csv')
                    print(corr_data_best)


            model.eval()
            attack_iters, n_restarts = (50, 1) if not args.debug else (10, 1)
            train_err, train_loss, _ = rob_err(train_batches_fast, model, eps, pgd_alpha, opt, half_prec, 0, 1)

            acc_corrs = []
  
            logger.info(
                '[train] err {:.2%}, loss {:.3f}'
                .format(train_err, train_loss)
            )
            logger.info('[test]  err {:.2%}'.format(test_err))
            if args.dataset == 'cifar10':
                for j in range(5):
                    acc_corr = clean_accuracy(model, x_corrs_fast[j].cuda(), y_corrs_fast[j].cuda())
                    acc_corrs.append(acc_corr)

                corr_err = 1 - np.mean(np.array(acc_corrs))

                logger.info('[corr]  err {:.2%}'.format(corr_err))

            if not args.debug:
                np.save('exps/{}.npy'.format(model_name), metr_dict)

            if args.final_adv_eval and epoch == args.epochs:
                context_manager = amp.disable_casts() if half_prec else utils.nullcontext()
                with context_manager:
                    last_state_dict = copy.deepcopy(model.state_dict())
                    half_prec = False
                    model.load_state_dict(last_state_dict)
                    model.eval()
                    opt = torch.optim.SGD(model.parameters(), lr=0)

                    test_err, _, _ = rob_err(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                    x_eval, y_eval, _ = data.get_xy_from_loader(test_batches)
                    adversary = autoattack.AutoAttack(model, norm='Linf', eps=eps)
                    x_eval_adv = adversary.run_standard_evaluation(x_eval, y_eval, bs=args.batch_size_eval)
                    test_err_rob = (model(x_eval_adv).argmax(1) == y_eval).float().mean().item()
                    logger.info('[last: test on {} points] err {:.2%}, err_rob {:.2%}'.format(args.n_final_eval, test_err, test_err_rob))
                    metr_dict['test_err_final'], metr_dict['test_err_pgd_rr_final'] = test_err, test_err_rob

                    if args.eval_early_stopped_model:
                        model.load_state_dict(best_state_dict)
                        model.eval()
                        test_err, _, _ = rob_err(test_batches, model, eps, pgd_alpha, opt, half_prec, 0, 1)
                        x_eval, y_eval, _ = data.get_xy_from_loader(test_batches)
                        adversary = autoattack.AutoAttack(model, norm='Linf', eps=eps)
                        x_eval_adv = adversary.run_standard_evaluation(x_eval, y_eval, bs=args.batch_size_eval)
                        test_err_rob = (model(x_eval_adv).argmax(1) == y_eval).float().mean().item()
                        logger.info('[best: test on {} points][iter={}] err {:.2%}, err_rob {:.2%}'.format(
                            args.n_final_eval, best_iteration, test_err, test_err_rob))
                        metr_dict['test_err_final_best'], metr_dict['test_err_pgd_rr_final_best'] = test_err, test_err_rob

            if not args.debug:
                np.save('exps/{}.npy'.format(model_name), metr_dict)
            model.train()

    logger.info('Done in {:.2f}m'.format((time.time() - start_time) / 60))

    if args.debug:
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    main()
