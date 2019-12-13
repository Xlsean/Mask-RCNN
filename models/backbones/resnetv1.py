########################################################################

# -*- coding: utf-8 -*-

########################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn

########################################################################


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    '''
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1,
              bias=True, padding_mode='zeros')

    nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False)

    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True)
    '''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    '''
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1,
              bias=True, padding_mode='zeros')

    nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False)

    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True)
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        residual = x

        if self.downsample is not None:
            residual = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += residual
        x = self.relu(x)

        return x


########################################################################


class ResNet(nn.Module):
    '''
    nn.Conv2d(in_channels, out_channels, kernel_size,
              stride=1, padding=0, dilation=1, groups=1,
              bias=True, padding_mode='zeros')

    nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False)

    nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1,
                   affine=True, track_running_stats=True)
    '''

    def __init__(self, bottleneck=True, inplanes=64, head7x7=True,
                 layers=(3, 4, 23, 3), num_classes=1000):
        super(ResNet, self).__init__()
        if bottleneck:
            block = Bottleneck
        else:
            block = BasicBlock

        self.inplanes = inplanes  # default 64
        self.head7x7 = head7x7

        if self.head7x7:
            self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7,
                                   stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(inplanes)
        else:
            self.conv1 = nn.Conv2d(3, inplanes // 2, kernel_size=3,
                                   stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(inplanes // 2)
            self.conv2 = nn.Conv2d(inplanes // 2, inplanes // 2, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(inplanes // 2)
            self.conv3 = nn.Conv2d(inplanes // 2, inplanes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, x):
        if self.head7x7:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)    # (Batch, 64, H, W)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)

            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)    # (Batch, 64, H, W)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def logits(self, x):
        x = self.avg_pool(x)    # (Batch, 2048, 1, 1)
        # Mind that the input image can not be larger than (416, 416),
        # or the output "x" would be (Batch, 2048, 2, 2)

        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)

        return x


########################################################################


def resnet18(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=False, inplanes=inplanes,
                  head7x7=head7x7, layers=(2, 2, 2, 2),
                  num_classes=num_classes)


def resnet26(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=True, inplanes=inplanes,
                  head7x7=head7x7, layers=(2, 2, 2, 2),
                  num_classes=num_classes)


def resnet34(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=False, inplanes=inplanes,
                  head7x7=head7x7, layers=(3, 4, 6, 3),
                  num_classes=num_classes)


def resnet50(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=True, inplanes=inplanes,
                  head7x7=head7x7, layers=(3, 4, 6, 3),
                  num_classes=num_classes)


def resnet101(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=True, inplanes=inplanes,
                  head7x7=head7x7, layers=(3, 4, 23, 3),
                  num_classes=num_classes)


def resnet152(inplanes=64, head7x7=True, num_classes=1000):
    return ResNet(bottleneck=True, inplanes=inplanes,
                  head7x7=head7x7, layers=(3, 8, 36, 3),
                  num_classes=num_classes)


########################################################################

if __name__ == '__main__':
    print(resnet50())
    

