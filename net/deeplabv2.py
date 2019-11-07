#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem
from .resnet50 import FixedBatchNorm, Bottleneck

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """
    DeepLab v2: Dilated ResNet + ASPP
    Output stride is fixed at 8
    """

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        # TODO: replace with standard resnet50
        ch = [64 * 2 ** p for p in range(6)]
        # self.add_module("layer1", _Stem(ch[0]))
        # self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        # self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        # self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        # self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        block = Bottleneck
        self.inplanes = 64
        self.add_module('conv1', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.add_module('bn1', FixedBatchNorm(64))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.add_module('layer1', self._make_layer(block, 64, n_blocks[0], stride=1, dilation=1))
        self.add_module('layer2', self._make_layer(block, 128, n_blocks[1], stride=2, dilation=1))
        self.add_module('layer3', self._make_layer(block, 256, n_blocks[2], stride=1, dilation=2))
        self.add_module('layer4', self._make_layer(block, 512, n_blocks[3], stride=1, dilation=4))
        self.inplanes = 1024

        self.add_module("aspp", _ASPP(ch[5], n_classes, atrous_rates))
    
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                FixedBatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forwardMSF(self, x_pack): # x_pack[0] [16, 2, 3, 512, 512]
        def flip_add(inp):
            return (inp[:,0]+inp[:,1].flip(-1))/2
        def fiveD_forward(inp):
            N = inp.shape[0]
            out = self.forward(inp.view(N*2,*(inp.shape[2:])))
            out = out.view(N,2,*(out.shape[1:]))
            return out
        # size = x_pack[0].shape[-2:]
        # strided_size = imutils.get_strided_size(size, 16)
        outputs = [flip_add(fiveD_forward(img))
                       for img in x_pack]
        strided_size = outputs[0].shape[-2:]
        strided_cam = torch.sum(torch.stack(
            [F.interpolate(o, strided_size, mode='bilinear', align_corners=False) for o
                in outputs]), 0)
        return strided_cam

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()

    def get_params(self, key):
        # For Dilated FCN
        if key == "1x":
            for m in self.named_modules():
                if "aspp" not in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            yield p
        # For conv weight in the ASPP module
        if key == "10x":
            for m in self.named_modules():
                if "aspp" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].weight
        # For conv bias in the ASPP module
        if key == "20x":
            for m in self.named_modules():
                if "aspp" in m[0]:
                    if isinstance(m[1], nn.Conv2d):
                        yield m[1].bias


class MSC(nn.Module):
    """
    Multi-scale inputs
    """

    def __init__(self, base, scales=None):
        super(MSC, self).__init__()
        self.base = base
        if scales:
            self.scales = scales
        else:
            self.scales = [0.5, 0.75]

    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max

def DeepLabV2_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )

def DeepLabV2_ResNet50_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 6, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )

if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)