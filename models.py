import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    def __init__(self, mu, std):
        super(Normalize, self).__init__()
        self.mu, self.std = mu, std

    def forward(self, x):
        return (x - self.mu) / self.std


class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
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

    def __init__(self, in_planes, planes, bn, learnable_bn, stride=1):
        super(PreActBlock, self).__init__()
        self.collect_preact = True
        self.avg_preacts = []
        self.bn1 = nn.BatchNorm2d(in_planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=not learnable_bn)
        self.bn2 = nn.BatchNorm2d(planes, affine=learnable_bn) if bn else IdentityLayer()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=not learnable_bn)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=not learnable_bn)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x  # Important: using out instead of x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet_Imagenet(nn.Module):
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, cifar_norm=True):
        super(PreActResNet_Imagenet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        if cifar_norm:
            self.mu = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()
        
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
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride))
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
    def __init__(self, block, num_blocks, n_cls, model_width=64, cuda=True, cifar_norm=True):
        super(PreActResNet, self).__init__()
        self.bn = True
        self.learnable_bn = True  # doesn't matter if self.bn=False
        self.in_planes = model_width
        self.avg_preact = None
        if cifar_norm:
            self.mu = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.2471, 0.2435, 0.2616)).view(1, 3, 1, 1)
        else:
            self.mu = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(1, 3, 1, 1)

        if cuda:
            self.mu = self.mu.cuda()
            self.std = self.std.cuda()

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
            layers.append(block(self.in_planes, planes, self.bn, self.learnable_bn, stride))
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


def PreActResNet18(n_cls, model_width=64, cuda=True, cifar_norm=True):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda, cifar_norm=cifar_norm)


def PreActResNet18_I(n_cls, model_width=64, cuda=True, cifar_norm=True):
    return PreActResNet_Imagenet(PreActBlock, [2, 2, 2, 2], n_cls=n_cls, model_width=model_width, cuda=cuda, cifar_norm=cifar_norm)


def PreActResNet50(n_cls, model_width=64, cuda=True, cifar_norm=True):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, cifar_norm=cifar_norm)


def PreActResNet50_I(n_cls, model_width=64, cuda=True, cifar_norm=True):
    return PreActResNet_Imagenet(PreActBottleneck, [3, 4, 6, 3], n_cls=n_cls, model_width=model_width, cuda=cuda, cifar_norm=cifar_norm)


def get_layers_dict(model):
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
