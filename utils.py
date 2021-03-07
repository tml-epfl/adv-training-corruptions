import logging
import numpy as np
import torch
import torch.nn.functional as F


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


def sign(grad):
    grad_sign = torch.sign(grad)
    return grad_sign


def attack_pgd_training(model, X, y, eps, alpha, attack_iters, rs=False, early_stopping=False, dist='linf'):
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


def attack_pgd(model, X, y, eps, alpha, attack_iters, n_restarts, rs=True, verbose=False,
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


def rob_err(batches, model, eps, pgd_alpha, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, corner=False, verbose=False, cuda=True):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []
    for i, (X, y) in enumerate(batches):
        if cuda:
            X, y = X.cuda(), y.cuda()
        pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, attack_iters, n_restarts, rs=rs,
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

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    return 1 - robust_acc, avg_loss, pgd_delta_np


def get_logits(batches, model, eps, pgd_alpha, attack_iters, adversarial=True):
    x_list, logits_list = [], []
    for i, (X, y, ln) in enumerate(batches):
        X, y = X.cuda(), y.cuda()
        if adversarial:
            pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, attack_iters, 1)
        else:
            pgd_delta = torch.zeros_like(X)

        with torch.no_grad():
            logits = model(X + pgd_delta)
            x_list.append((X+pgd_delta).cpu())
            logits_list.append(logits.cpu())
    x_all = torch.cat(x_list)
    logits_all = torch.cat(logits_list)
    return x_all, logits_all


def get_clean_pred(batches, model):
    logits_list = []
    for i, (X, y) in enumerate(batches):
        with torch.no_grad():
            X, y = X.cuda(), y.cuda()
            logits_batch = model(X)
            logits_list.append(logits_batch.cpu().numpy())

    logits = np.vstack(logits_list)
    return logits


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


def update_metrics(metrics_dict, metrics_values, metrics_names):
    assert len(metrics_values) == len(metrics_names)
    for metric_value, metric_name in zip(metrics_values, metrics_names):
        metrics_dict[metric_name].append(metric_value)
    return metrics_dict
