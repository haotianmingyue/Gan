#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/9/22
# @file ss_gan.py

import torch
import torch.nn as nn


def x2conv(in_channels, out_channels, inner_channels=None):
    # when inner_channels is None it shows that only one conv
    if inner_channels is None:
        down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    else:
        pass

    return down_conv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.rs_con = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x1 = self.down_conv(x)
        # x1 = self.pool(x1)
        x2 = self.rs_con(x)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        pass

    def forward(self, z, labels):
        # shape of z: [batchsize, latent_dim]

       pass



class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        pass

    def forward(self, image, labels):
        # shape of images: [batchsize, 1, 28, 28]

        pass


