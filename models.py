import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class CustomReLU(nn.Module):
    def __init__(self):
        super(CustomReLU, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []

    def forward(self, preact):
        if self.collect_preact:
            self.avg_preacts.append(preact.abs().mean().item())
        act = F.relu(preact)
        return act


class ModuleWithStats(nn.Module):
    def __init__(self):
        super(ModuleWithStats, self).__init__()

    def forward(self, x):
        for layer in self._model:
            if type(layer) == CustomReLU:
                layer.avg_preacts = []

        out = self._model(x)

        avg_preacts_all = [layer.avg_preacts for layer in self._model if type(layer) == CustomReLU]
        self.avg_preact = np.mean(avg_preacts_all)
        return out


class Linear(ModuleWithStats):
    def __init__(self, n_cls, shape_in):
        super(Linear, self).__init__()
        d = int(np.prod(shape_in[1:]))
        self._model = nn.Sequential(
            Flatten(),
            nn.Linear(d, n_cls)
        )


class FC(ModuleWithStats):
    def __init__(self, n_cls, shape_in, n_hl, n_hidden):
        super(FC, self).__init__()
        fc_layers = []
        for i_layer in range(n_hl):
            n_in = np.prod(shape_in[1:]) if i_layer == 0 else n_hidden
            n_out = n_hidden
            fc_layers += [nn.Linear(n_in, n_out), CustomReLU()]
        self._model = nn.Sequential(
            Flatten(),
            *fc_layers,
            CustomReLU(),
            nn.Linear(n_hidden, n_cls)
        )


class CNNBase(ModuleWithStats):
    def __init__(self):
        super(CNNBase, self).__init__()


class CNN(CNNBase):
    def __init__(self, n_cls, shape_in, n_conv, n_filters):
        super(CNN, self).__init__()
        input_size = shape_in[2]
        conv_blocks = []
        for i_layer in range(n_conv):
            n_in = shape_in[1] if i_layer == 0 else n_filters
            n_out = n_filters
            conv_blocks += [nn.Conv2d(n_in, n_out, 3, stride=1, padding=1), CustomReLU()]
        # h_after_conv, w_after_conv = input_size/n_conv, input_size/n_conv
        h_after_conv, w_after_conv = input_size, input_size
        self._model = nn.Sequential(
            *conv_blocks,
            Flatten(),
            nn.Linear(n_filters*h_after_conv*w_after_conv, n_cls)  # a bit large, but ok (163840 parameters for 16 filters)
        )


class CNNLeNet(CNNBase):
    def __init__(self, n_cls, shape_in):
        super(CNNLeNet, self).__init__()
        self._model = nn.Sequential(
            nn.Conv2d(shape_in[1], 16, 4, stride=2, padding=1),
            CustomReLU(),
            # nn.Dropout2d(p=0.5),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            CustomReLU(),
            Flatten(),
            # nn.Dropout(p=0.5),
            nn.Linear(32*7*7, 100),
            CustomReLU(),
            nn.Linear(100, n_cls)
        )


class CNNLeNetGAP(CNNBase):
    def __init__(self, n_cls, shape_in):
        super(CNNLeNetGAP, self).__init__()
        self._model = nn.Sequential(
            nn.Conv2d(shape_in[1], 16, 4, stride=2, padding=1),
            CustomReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            CustomReLU(),
            # Flatten(),
            nn.AvgPool2d((7, 7)),  # global average pooling
            Flatten(),
            nn.Linear(32, 100),
            CustomReLU(),
            nn.Linear(100, n_cls)
        )


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out



class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1, activation='relu'):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.activation = activation
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def act_function(self, preact):
        if self.activation == 'relu':
            act = F.relu(preact)
        else:
            assert self.activation[:8] == 'softplus'
            beta = int(self.activation.split('softplus')[1])
            act = F.softplus(preact, beta=beta)
        return act

    def forward(self, x):
        out = self.act_function(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(self.act_function(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet_Imagenet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
        super(PreActResNet_Imagenet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        self.activation = activation
        #print(num_blocks)
        if cifar_norm:
            self.mu = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(model_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, model_width, kernel_size=7, stride=2, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*model_width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*model_width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*model_width, num_blocks[3], stride=2)
        self.bn2 = nn.BatchNorm2d(8*model_width * block.expansion)
        self.linear = nn.Linear(8*model_width*block.expansion, n_cls)
        #print(self.linear)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, delta=None, ri=-1):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []
        
        if delta is None:
            out = self.normalize(x)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            #print(out.shape)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = F.relu(self.bn2(out))

            out = self.avgpool(out)

            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif ri == -1:
            out = self.normalize(torch.clamp(x + delta[0], 0, 1))
            out = self.conv1(out) + delta[1]
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)

            out = self.layer1(out)  + delta[2]
            out = self.layer2(out)  + delta[3]
            out = self.layer3(out)  + delta[4]
            out = self.layer4(out)  + delta[5]
            
            out = F.relu(self.bn2(out))

            out = self.avgpool(out)
            
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        self.activation = activation
        if cifar_norm:
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        if half_prec:
            self.mu = self.mu.half()
            self.std = self.std.half()

        self.normalize = Normalize(self.mu, self.std)
        self.conv1 = nn.Conv2d(3, model_width, kernel_size=3, stride=1, padding=1, bias=not self.learnable_bn)
        self.layer1 = self._make_layer(block, model_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2*model_width, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*model_width, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*model_width, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(8*model_width * block.expansion)
        self.linear = nn.Linear(8*model_width*block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride, self.activation))
            # layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, delta=None, ri=-1):
        for layer in [*self.layer1, *self.layer2, *self.layer3, *self.layer4]:
            layer.avg_preacts = []
        
        if delta is None:
            out = self.normalize(x)
            out = self.conv1(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif ri == -1:
            out = self.normalize(torch.clamp(x + delta[0], 0, 1))
            out = self.conv1(out) + delta[1]
            out = self.layer1(out)  + delta[2]
            out = self.layer2(out)  + delta[3]
            out = self.layer3(out)  + delta[4]
            out = self.layer4(out)  + delta[5]
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out) #+ delta[6]
        else:
            out = self.normalize(torch.clamp(x + delta[0], 0, 1)) if ri == 0 else self.normalize(torch.clamp(x, 0, 1))
            out = self.conv1(out + delta[1])  if ri == 1 else self.conv1(out)
            out = self.layer1(out + delta[2]) if ri == 2 else self.layer1(out)
            out = self.layer2(out + delta[3]) if ri == 3 else self.layer2(out)
            out = self.layer3(out + delta[4]) if ri == 4 else self.layer3(out)
            out = self.layer4(out + delta[5]) if ri == 5 else self.layer4(out)
            out = F.relu(self.bn(out))
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out + delta[6]) if ri == 6 else self.linear(out)

        return out


def PreActResNet18(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, cifar_norm=cifar_norm)

def PreActResNet18_I(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
    return PreActResNet_Imagenet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, cifar_norm=cifar_norm)


def PreActResNet50(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, cifar_norm=cifar_norm)

def PreActResNet50_I(n_cls, model_width=64, cuda=True, half_prec=False, activation='relu', cifar_norm=True):
    return PreActResNet_Imagenet(PreActBottleneck, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, half_prec=half_prec,
                        activation=activation, cifar_norm=cifar_norm)

def get_model(model_name, n_cls, half_prec, shapes_dict, model_width, n_filters_cnn, n_hidden_fc, activation, cifar_norm=True):
    if model_name == 'resnet18':
        if shapes_dict[-1] == 32:
            model = PreActResNet18(n_cls, model_width=model_width, half_prec=half_prec, activation=activation, cifar_norm=cifar_norm)
        elif shapes_dict[-1] == 224:
            model = PreActResNet18_I(n_cls, model_width=model_width, half_prec=half_prec, activation=activation, cifar_norm=cifar_norm)
    elif model_name == 'resnet50':
        if shapes_dict[-1] == 32:
            model = PreActResNet50(n_cls, model_width=model_width, half_prec=half_prec, activation=activation, cifar_norm=cifar_norm)
        elif shapes_dict[-1] == 224:
            model = PreActResNet50_I(n_cls, model_width=model_width, half_prec=half_prec, activation=activation, cifar_norm=cifar_norm)

    elif model_name == 'lenet':
        model = CNNLeNet(n_cls, shapes_dict)
    elif model_name == 'cnn':
        model = CNN(n_cls, shapes_dict, 1, n_filters_cnn)
    elif model_name == 'fc':
        model = FC(n_cls, shapes_dict, 1, n_hidden_fc)
    elif model_name == 'linear':
        model = Linear(n_cls, shapes_dict)
    else:
        raise ValueError('wrong model')
    return model


def get_models_dict(model):
    return {
        'default': [model.normalize,
                    model.conv1,
                    model.layer1,
                    model.layer2,
                    model.layer3,
                    model.layer4],
        'lpips': [model.conv1,
                  model.layer1,
                  model.layer2,
                  model.layer3,
                  model.layer4],
        'all': [model.normalize,
                model.conv1,
                model.layer1[0].bn1,
                model.layer1[0].conv1,
                model.layer1[0].bn2,
                model.layer1[0].conv2,
                model.layer1[1].bn1,
                model.layer1[1].conv1,
                model.layer1[1].bn2,
                model.layer1[1].conv2,
                model.layer1,
                model.layer2[0].bn1,
                model.layer2[0].conv1,
                model.layer2[0].bn2,
                model.layer2[0].conv2,
                model.layer2[1].bn1,
                model.layer2[1].conv1,
                model.layer2[1].bn2,
                model.layer2[1].conv2,
                model.layer2,
                model.layer3[0].bn1,
                model.layer3[0].conv1,
                model.layer3[0].bn2,
                model.layer3[0].conv2,
                model.layer3[1].bn1,
                model.layer3[1].conv1,
                model.layer3[1].bn2,
                model.layer3[1].conv2,
                model.layer3,
                model.layer4[0].bn1,
                model.layer4[0].conv1,
                model.layer4[0].bn2,
                model.layer4[0].conv2,
                model.layer4[1].bn1,
                model.layer4[1].conv1,
                model.layer4[1].bn2,
                model.layer4[1].conv2,
                model.layer4, ],
        'bnonly': [model.normalize,
                   model.layer1[0].bn1,
                   model.layer1[0].bn2,
                   model.layer1[1].bn1,
                   model.layer1[1].bn2,
                   model.layer2[0].bn1,
                   model.layer2[0].bn2,
                   model.layer2[1].bn1,
                   model.layer2[1].bn2,
                   model.layer3[0].bn1,
                   model.layer3[0].bn2,
                   model.layer3[1].bn1,
                   model.layer3[1].bn2,
                   model.layer4[0].bn1,
                   model.layer4[0].bn2,
                   model.layer4[1].bn1,
                   model.layer4[1].bn2],
        'bn1only': [model.normalize,
                    model.layer1[0].bn1,
                    model.layer1[1].bn1,
                    model.layer2[0].bn1,
                    model.layer2[1].bn1,
                    model.layer3[0].bn1,
                    model.layer3[1].bn1,
                    model.layer4[0].bn1,
                    model.layer4[1].bn1],
        'convonly': [model.normalize,
                     model.conv1,
                     model.layer1[0].conv1,
                     model.layer1[0].conv2,
                     model.layer1[1].conv1,
                     model.layer1[1].conv2,
                     model.layer2[0].conv1,
                     model.layer2[0].conv2,
                     model.layer2[1].conv1,
                     model.layer2[1].conv2,
                     model.layer3[0].conv1,
                     model.layer3[0].conv2,
                     model.layer3[1].conv1,
                     model.layer3[1].conv2,
                     model.layer4[0].conv1,
                     model.layer4[0].conv2,
                     model.layer4[1].conv1,
                     model.layer4[1].conv2],
        'conv1only': [model.normalize,
                      model.conv1,
                      model.layer1[0].conv1,
                      model.layer1[1].conv1,
                      model.layer2[0].conv1,
                      model.layer2[1].conv1,
                      model.layer3[0].conv1,
                      model.layer3[1].conv1,
                      model.layer4[0].conv1,
                      model.layer4[1].conv1],
        'single': [model.layer2]
    }
