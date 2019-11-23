########################################################################

# -*- coding: utf-8 -*-

########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as fn

########################################################################


class BasicBlock(nn.Module):
    '''
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1,
              bias=True, padding_mode='zeros')

    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True)
    '''

    def __init__(self, inplanes, growth_rate, dropout=0.0):
        super(BasicBlock, self).__init__()
        self.DROPOUT = dropout
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes, growth_rate, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)

        if self.DROPOUT > 0.0:
            out = self.dropout(out)

        return torch.cat([x, out], dim=1)


class Bottleneck(nn.Module):
    '''
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1,
              bias=True, padding_mode='zeros')

    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True)
    '''

    def __init__(self, inplanes, growth_rate, dropout=0.0):
        super(Bottleneck, self).__init__()
        self.DROPOUT = dropout
        outplanes = growth_rate * 4

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv2 = nn.Conv2d(outplanes, growth_rate, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.DROPOUT > 0.0:
            out = self.dropout(out)

        return torch.cat([x, out], dim=1)


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(inplanes)
        self.conv = nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)
        return x


########################################################################


class DenseNet(nn.Module):
    """docstring for DenseNet"""

    def __init__(self, bottleneck=True, num_block=(6, 12, 24, 16),
                 growth_rate=12, reduction=0.5, dropout=0.2, num_classes=10):
        super(DenseNet, self).__init__()
        if bottleneck:
            block = Bottleneck
        else:
            block = BasicBlock

        self.growth_rate = growth_rate
        self.dropout = dropout

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, 3, 1, 1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, num_block[0])
        num_planes += num_block[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, num_block[1])
        num_planes += num_block[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, num_block[2])
        num_planes += num_block[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, num_block[3])
        num_planes += num_block[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(num_planes, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_dense_layers(self, block, in_planes, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(in_planes, self.growth_rate, self.dropout))
            in_planes += self.growth_rate

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.trans3(x)
        x = self.dense4(x)

        return x

    def logits(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)

        return x


########################################################################


def densenet121(num_classes=10):
    return DenseNet(bottleneck=True, num_block=[6, 12, 24, 16],
                    growth_rate=32, reduction=0.5, dropout=0.2,
                    num_classes=num_classes)


def densenet169(num_classes=10):
    return DenseNet(bottleneck=True, num_block=[6, 12, 32, 32],
                    growth_rate=32, reduction=0.5, dropout=0.2,
                    num_classes=num_classes)


def densenet201(num_classes=10):
    return DenseNet(bottleneck=True, num_block=[6, 12, 48, 32],
                    growth_rate=32, reduction=0.5, dropout=0.2,
                    num_classes=num_classes)


def densenet161(num_classes=10):
    return DenseNet(bottleneck=True, num_block=[6, 12, 36, 24],
                    growth_rate=48, reduction=0.5, dropout=0.2,
                    num_classes=num_classes)


def densenet_cifar(num_classes=10):
    return DenseNet(bottleneck=True, num_block=[6, 12, 24, 16],
                    growth_rate=12, reduction=0.5, dropout=0.2,
                    num_classes=num_classes)

########################################################################


if __name__ == '__main__':
    pass
