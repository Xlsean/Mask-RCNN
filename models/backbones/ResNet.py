############################################################################################################

# -*- coding: utf-8 -*-

'''
ResNet in Pytorch
Author: Xlsean
'''

############################################################################################################

import math
import numpy as np
import torch
import torch.nn as nn

############################################################################################################

class Bottleneck(nn.Module):
    '''
    The identity block is the standard block used in ResNets.
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        x_shortcut = x

        if self.downsample is not None:
            x_shortcut = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = x + x_shortcut
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    '''
    The implementation of ResNet50 & ResNet101.
    '''
    def __init__(self, architecture):
        super(ResNet, self).__init__()
        assert architecture in ["ResNet50", "ResNet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"ResNet50":6, "ResNet101":23}[architecture], 3]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2_x = self.make_layer(Bottleneck, 64, self.layers[0], stride=1)
        self.conv3_x = self.make_layer(Bottleneck, 128, self.layers[1], stride=2)
        self.conv4_x = self.make_layer(Bottleneck, 256, self.layers[2], stride=2)
        self.conv5_x = self.make_layer(Bottleneck, 512, self.layers[3], stride=2)

    def stages(self):
        return [self.conv1, self.conv2_x, self.conv3_x, self.conv4_x, self.conv5_x]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        return x
    
    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

############################################################################################################

if __name__ == '__main__':
    model = ResNet("ResNet50")
    print(model)