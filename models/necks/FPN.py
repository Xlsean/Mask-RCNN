############################################################################################################

# -*- coding: utf-8 -*-

'''
ResNet in Pytorch
Author: Xlsean
'''

############################################################################################################

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.abspath("./backbones"))
# print(sys.path)
from ResNet import *

############################################################################################################

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels

        # Bottom-up layers
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5

        # Top layer
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)

        # Smooth layers
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Bottom-up
        x = self.C1(x)
        x = self.C2(x)
        C2_out = x
        x = self.C3(x)
        C3_out = x
        x = self.C3(x)
        C4_out = x
        x = self.C3(x)
        C5_out = x

        # Top-down
        P5_out = self.latlayer1(C5_out)
        P4_out = self.latlayer2(C4_out) + F.upsample(P5_out, scale_factor=2)
        P3_out = self.latlayer3(C3_out) + F.upsample(P4_out, scale_factor=2)
        P2_out = self.latlayer4(C2_out) + F.upsample(P3_out, scale_factor=2)

        # Smooth
        P5 = self.smooth(P5_out)
        P4 = self.smooth(P4_out)
        P3 = self.smooth(P3_out)
        P2 = self.smooth(P2_out)
        P6 = self.P6(P5_out)

        return [P2, P3, P4, P5, P6]

############################################################################################################
if __name__ == '__main__':
    ResNet = ResNet("ResNet50")
    C1, C2, C3, C4, C5 = ResNet.stages()
    FPN = FPN(C1, C2, C3, C4, C5, out_channels=256)
    print(FPN)
