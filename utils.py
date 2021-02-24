import logging

import gc
import math
import apex.amp as amp
import numpy as np
import torch
import torch.nn.functional as F
import data
from contextlib import contextmanager
from torch import nn


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


# X_rand = torch.rand([200, 3, 32, 32]).cuda()
# X_rand = (torch.rand([200, 3, 32, 32]) > 0.5).float().cuda()
# idx_x = np.random.permutation(200*3*32*32)
# y_rand = torch.Tensor(np.random.randint(0, 10, size=200)).long().cuda()


def get_grad_np(model, batches, eps, opt, half_prec, rs=False, cross_entropy=True):
    grad_list = []
    for i, (X, y) in enumerate(batches):
        X, y = X.cuda(), y.cuda()
        # X = X.view(-1)[idx_x].view(200, 3, 32, 32)
        # X = X_rand
        # y = y_rand
        # X = X.mean(1, keepdim=True).repeat(1, 3, 1, 1)  # grayscale

        if rs:
            delta = get_uniform_delta(X.shape, eps, requires_grad=False)
        else:
            delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        logits = model(clamp(X + delta, 0, 1))

        if cross_entropy:
            loss = F.cross_entropy(logits, y)
        else:
            y_onehot = torch.zeros([len(y), 10]).long().cuda()
            y_onehot.scatter_(1, y[:, None], 1)
            preds_correct_class = (logits * y_onehot.float()).sum(1, keepdim=True)
            margin = preds_correct_class - logits  # difference between the correct class and all other classes
            margin += y_onehot.float() * 10000  # to exclude zeros coming from f_correct - f_correct
            margin = margin.min(1, keepdim=True)[0]
            loss = F.relu(1 - margin).mean()

        if half_prec:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
                delta.grad.mul_((loss / scaled_loss).item())
        else:
            loss.backward()
        grad = delta.grad.detach().cpu()
        # grad = torch.sign(grad)
        grad_list.append(grad.numpy())
        delta.grad.zero_()
    grads = np.vstack(grad_list)
    return grads


def get_input_grad(model, X, y, opt, eps, half_prec, delta_init='none', backprop=False):
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model(X + delta)  # TODO: experimental
    # output = model(clamp(X + delta, 0, 1))
    loss = F.cross_entropy(output, y)
    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            grad = torch.autograd.grad(scaled_loss, delta, create_graph=True if backprop else False)[0]
            grad /= scaled_loss / loss
    else:
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad


def calc_distances_hl1(model, eval_batches):
    distances_list = []
    for i, (X, y) in enumerate(eval_batches):
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            distances = model.calc_distances_hl1(X)
        distances_list.append(distances.cpu().numpy())
    distances = np.vstack(distances_list)
    return distances


def configure_logger(model_name, debug):
    logging.basicConfig(format='%(message)s')  # , level=logging.DEBUG)
    logger = logging.getLogger()
    logger.handlers = []  # remove the default logger

    # add a new logger for stdout
    formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if not debug:
        # add a new logger to a log file
        logger.addHandler(logging.FileHandler('logs/{}.log'.format(model_name)))

    return logger


def to_eval_halfprec(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval().half()


def to_train_halfprec(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train().float()


def to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def to_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def model_eval(model, half_prec):
    model.eval()
    # model = model.eval()
    # model.apply(to_eval)
    # if half_prec:
    #     model.apply(to_eval_halfprec)  # has to be disabled if learnable_bn=False because of an error
    # else:
    #     model.apply(to_eval)


def model_train(model, half_prec):
    model.train()
    # if half_prec:
    #     model.apply(to_train_halfprec)
    # else:
    #     model.apply(to_train)

def get_output_of_layers(model, X, layers):
    outputs = []
    def out_hook(m, inp, out):
        outputs.append(out)
    
    handles = [layer.register_forward_hook(out_hook) for layer in layers]
    out = model(X)

    for handle in handles:
        handle.remove()
    return outputs


def get_input_of_layers(model, X, layers):
    outputs = []
    def out_hook(m, inp, out):
        outputs.append(inp[0])
    
    handles = [layer.register_forward_hook(out_hook) for layer in layers]
    out = model(X)

    for handle in handles:
        handle.remove()
    return outputs

def prune_conv_layer(model, id_layer, prune_frac):
    """ prunes `prune_frac`% smallest values """
    prune_frac = min(prune_frac, 1.0)  # if prune_frac happens to exceed 1.0
    with torch.no_grad():
        param = list(model.parameters())[id_layer]
        retain_frac = 1 - prune_frac
        k = int(retain_frac * np.prod(param.shape))
        topk_values = torch.topk(param.flatten().abs(), k=k, largest=True, sorted=False)[0]
        kth_value = topk_values.min() if topk_values.shape[0] > 0 else 0.0
        binary_mask = (param.abs() > kth_value).type(param.data.type()).cuda()
        param.data = param.data * binary_mask


def get_uniform_delta(shape, eps, requires_grad=True, dist='linf'):
    delta = torch.zeros(shape).cuda()
    if dist == 'l2':
        delta.normal_(mean=0, std=1.0)
        r = np.random.uniform(0, eps)
        delta.data = delta.data * r / (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
    elif dist == 'linf':
        delta.uniform_(-eps, eps)
        delta.requires_grad = requires_grad
    return delta


def get_gaussian_delta(shape, eps, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    delta = eps * torch.randn(*delta.shape)
    delta.requires_grad = requires_grad
    return delta


def get_cuda_tensors():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    return cuda_tensors


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def attack_pgd_training(model, X, y, eps, alpha, opt, half_prec, attack_iters, rs=False, early_stopping=False, dist='linf'):
    delta = torch.zeros_like(X).cuda()
    if rs:
        if dist == 'l2':
            delta.normal_(mean=0, std=1.0)
            r = np.random.uniform(0, eps)
            delta.data = delta.data * r / (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
        elif dist == 'linf':
            delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(clamp(X + delta, 0, 1))
        loss = F.cross_entropy(output, y)
        if half_prec:
            with amp.scale_loss(loss, opt) as scaled_loss:
                # delta.grad = torch.autograd.grad(scaled_loss, delta)[0]
                scaled_loss.backward()
                delta.grad.mul_(loss.item() / scaled_loss.item())
        else:
            #print(loss.shape)
            loss.backward()
        grad = delta.grad.detach()

        if early_stopping:
            # stabilization trick for MNIST (eps=0.3) from Wong et al, ICLR'20; without it converges to 10% accuracy
            # alternatively: larger model size also helps to prevent this
            idx_update = output.max(1)[1] == y
        else:
            idx_update = torch.ones(y.shape, dtype=torch.bool)
        
        if dist == 'l2':
            grad_norm = (grad**2).sum([1, 2, 3], keepdim=True)**0.5
            delta.data = delta + alpha * grad / grad_norm
            delta.data = clamp(X + delta.data, 0, 1) - X
            delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
            delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()
        else:
            grad_sign = sign(grad)
            delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
            delta.data = clamp(X + delta.data, 0, 1) - X
            delta.data = clamp(delta.data, -eps, eps)
            delta.grad.zero_()

    return delta.detach()


def attack_pgd(model, X, y, eps, alpha, opt, half_prec, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)  # + 0.25*torch.randn(X.shape).cuda())  # adding noise (aka smoothing)
            loss = F.cross_entropy(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    # delta.grad = torch.autograd.grad(scaled_loss, delta)[0]
                    scaled_loss.backward()
                    delta.grad.mul_(loss.item() / scaled_loss.item())
            else:
                loss.backward()
            grad = delta.grad.detach()
            if not l2_grad_update:
                delta.data = delta + alpha * sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)
            delta.grad.zero_()

        with torch.no_grad():
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta

def eval_dataset(batches, model, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()

        with torch.no_grad():
            output = model(X)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)


    test_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    torch.cuda.empty_cache()
    return 1 - test_acc, avg_loss

def rob_err(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, print_fosc=False, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, n_restarts, rs=rs,
                               verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        if corner:
            pgd_delta = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X
        pgd_delta_proj = clamp(X + eps * sign(pgd_delta), 0, 1, cuda) - X  # needed just for investigation

        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())
        pgd_delta_proj_list.append(pgd_delta_proj.cpu().numpy())

        if print_fosc:
            pgd_delta.requires_grad = True
            output = model(X + pgd_delta)
            loss = F.cross_entropy(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                    pgd_delta.grad.mul_((loss / scaled_loss).item())
            else:
                loss.backward()
            grad, pgd_delta = grad.detach(), pgd_delta.detach()
            # pgd_delta_proj.abs() gives the final magnitude of the box constraints (including [0,1]^d)
            fosc_per_point = (pgd_delta_proj.abs() * grad.abs()).sum((1, 2, 3)) - (pgd_delta * grad).sum((1, 2, 3))
            min_fosc = torch.min(fosc_per_point)
            avg_fosc = torch.mean(fosc_per_point)
            max_fosc = torch.max(fosc_per_point)
            print('fosc: avg={:.5f} (min={:.5f}, max={:.5f})'.format(avg_fosc, min_fosc, max_fosc))

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    # pgd_delta_proj_np = np.vstack(pgd_delta_proj_list)
    # if n_restarts > 1:
    #     l2_pgd_delta = ((pgd_delta_np ** 2).sum((1, 2, 3)) ** 0.5)
    #     l2_pgd_delta_proj = ((pgd_delta_proj_np ** 2).sum((1, 2, 3)) ** 0.5)
    #     l2_diff = (((pgd_delta_np - pgd_delta_proj_np) ** 2).sum((1, 2, 3)) ** 0.5)
    #     inner_product = (pgd_delta_np * pgd_delta_proj_np).sum((1, 2, 3))
    #     cos = inner_product / (l2_pgd_delta * l2_pgd_delta_proj)
    #     n_zeros = (pgd_delta_np == 0.0).mean((1, 2, 3))
    #     print('delta vs delta_proj: l2_pgd_delta {:.3f}, l2_pgd_delta_proj {:.3f}, l2_diff {:.3f}, cos {:.3f}, inner_product {:.3f}, n_zeros {:.3%}'.format(
    #         l2_pgd_delta.mean(), l2_pgd_delta_proj.mean(), l2_diff.mean(), cos.mean(), inner_product.mean(), n_zeros.mean()))
    return 1 - robust_acc, avg_loss, pgd_delta_np


def get_logits(batches, model, eps, pgd_alpha, opt, half_prec, attack_iters, adversarial=True):
    x_list, logits_list = [], []
    for i, (X, y, ln) in enumerate(batches):
        X, y = X.cuda(), y.cuda()
        if adversarial:
            pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, opt, half_prec, attack_iters, 1)
        else:
            pgd_delta = torch.zeros_like(X)

        with torch.no_grad():
            logits = model(X + pgd_delta)
            x_list.append((X+pgd_delta).cpu())
            logits_list.append(logits.cpu())
    x_all = torch.cat(x_list)
    logits_all = torch.cat(logits_list)
    return x_all, logits_all


def rob_acc_corner_sampling(batches, model, eps, opt, half_prec, n_samples, corner=True, verbose=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list = []
    for i, (X, y) in enumerate(batches):
        X, y = X.cuda(), y.cuda()

        max_loss = torch.zeros(y.shape[0]).cuda()
        max_delta = torch.zeros_like(X).cuda()
        grad_norms_batch = []
        for i_sample in range(n_samples):
            delta = torch.zeros_like(X).cuda()
            delta.uniform_(-eps, eps)
            if corner:
                delta = eps * sign(delta)
            delta.requires_grad = True

            output = model(clamp(X + delta, 0, 1))
            loss = F.cross_entropy(output, y)
            if half_prec:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            grad = delta.grad.detach()
            delta = delta.detach()
            grad_norms = (torch.sum(grad ** 2, (1, 2, 3)) ** 0.5).cpu().numpy()
            grad_norms_batch.append(grad_norms)

            all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
            max_loss = torch.max(max_loss, all_loss)
            delta.grad.zero_()

            if verbose and n_samples > 1 and i_sample % 20 == 0:
                print('Restart #{}: max loss {:.3f}'.format(i_sample, max_loss.mean()))

        grad_norms = np.vstack(grad_norms_batch).T
        print('grad norms: min={:.4f}, avg={:.4f}, max={:.4f}'.format(grad_norms.min(1).mean(), grad_norms.mean(), grad_norms.max(1).mean()))

        delta = clamp(X + max_delta, 0, 1) - X
        with torch.no_grad():
            output = model(X + delta)
            loss = F.cross_entropy(output, y)
        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(delta.cpu().numpy())
    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex

    return robust_acc, avg_loss


def get_clean_pred(batches, model):
    logits_list = []
    for i, (X, y) in enumerate(batches):
        with torch.no_grad():
            X, y = X.cuda(), y.cuda()
            logits_batch = model(X)
            logits_list.append(logits_batch.cpu().numpy())

    logits = np.vstack(logits_list)
    return logits


def model_params_to_list(model):
    list_params = []
    model_params = list(model.parameters())
    for param in model_params:
        list_params.append(param.data.clone())
    return list_params


def avg_cos_np(v1, v2):
    # v1[v2 == 0] = 0
    # v2[v1 == 0] = 0
    norms1 = np.sum(v1 ** 2, (1, 2, 3), keepdims=True) ** 0.5
    norms2 = np.sum(v2 ** 2, (1, 2, 3), keepdims=True) ** 0.5
    cos_vals = np.sum(v1/norms1 * v2/norms2, (1, 2, 3))
    cos_vals[np.isnan(cos_vals)] = 1.0  # to prevent nans (0/0)
    cos_vals[np.isinf(cos_vals)] = 1.0  # to prevent +infs and -infs (x/0, -x/0)
    avg_cos = np.mean(cos_vals)
    return avg_cos


def avg_l2_np(v1, v2=None):
    if v2 is not None:
        diffs = v1 - v2
    else:
        diffs = v1
    diff_norms = np.sum(diffs ** 2, (1, 2, 3)) ** 0.5
    avg_norm = np.mean(diff_norms)
    return avg_norm


def avg_fraction_same_sign(v1, v2):
    v1 = np.sign(v1)
    v2 = np.sign(v2)
    avg_cos = np.mean(v1 == v2)
    return avg_cos


def l2_norm_batch(v):
    norms = (v ** 2).sum([1, 2, 3]) ** 0.5
    # norms[norms == 0] = np.inf
    return norms


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.in_features
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    # From Rice et al.: leads to a smaller loss at init, but it still diverges
    # m = module
    # if isinstance(m, nn.Conv2d):
    #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # elif isinstance(m, nn.BatchNorm2d):
    #     m.weight.data.fill_(1)
    #     m.bias.data.zero_()
    # elif isinstance(m, nn.Linear):
    #     m.bias.data.zero_()


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
    elif lr_schedule_type == 'piecewise':
        def lr_schedule(t):
            if n_epochs == 0:
                return lr_max
            if t / n_epochs < 0.34:#0.6:
                return lr_max
            elif t / n_epochs < 0.67:#< 0.9:
                return lr_max / 10.
            else:
                return lr_max / 100.
    else:
        raise ValueError('wrong lr_schedule_type')
    return lr_schedule


def backward(loss, opt, half_prec):
    if half_prec:
        with amp.scale_loss(loss, opt) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()


def update_metrics(metrics_dict, metrics_values, metrics_names):
    assert len(metrics_values) == len(metrics_names)
    for metric_value, metric_name in zip(metrics_values, metrics_names):
        metrics_dict[metric_name].append(metric_value)
    return metrics_dict


def calc_lin_approx_err(model, loader, eps, point='fgsm'):
    # TODO: currently supports only fp32 (fp16 may have undesirable effects due to the lack of the loss scaling)
    X, y = data.get_xy_from_loader(loader)
    X.requires_grad = True

    if point == 'fgsm':
        loss_x = F.cross_entropy(model(X), y, reduction='none')
        grads_x = torch.autograd.grad(loss_x.sum(), X)[0]
        X = X.detach()
        delta = eps * torch.sign(grads_x)
        loss_x_delta = F.cross_entropy(model(X + delta), y, reduction='none')
        linear_approx = loss_x + (delta * grads_x).sum([1, 2, 3])
    else:
        raise ValueError('unsupported point type to calculate the linear approx error')

    lin_approx_err = torch.abs(loss_x_delta - linear_approx).mean().item()
    return lin_approx_err


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

