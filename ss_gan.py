#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/9/22
# @file ss_gan.py

import torch
import torch.nn as nn


def x2conv(in_channels, out_channels, inner_channels=None):
    # when inner_channels is None it shows that only one conv
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding='same', bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    return down_conv


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.down_conv = x2conv(in_channels, out_channels)
        # self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

        # 用带步长的卷积而不用 池化操作
        self.rs_con = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.down_conv(x)
        x2 = self.rs_con(x)
        x3 = x1 + x2
        return x3

class Encoder_ss(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_ss, self).__init__()

        # 他结构图上没有池化，所以推测是用的步长为2的卷积来降低分辨率的，但他又来个
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.rs_con = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x1 = self.down_conv(x)
        x2 = self.rs_con(x)

        return x1 + x2


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels//2, in_channels//2, kernel_size=1, padding='same', bias=False)
        # self.conv11 = nn.Conv2d(1, 1, kernel_size=1, padding='same', bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.up_conv = x2conv(in_channels, out_channels)

    def forward(self, x_copy, x):
        x = self.up(x)

        # attention module
        x_copy1 = self.conv1(x_copy)
        x11 = self.conv1(x)
        n_x = x_copy1 + x11
        n_x = self.relu(n_x)
        n_x = self.conv1(n_x)
        n_x = self.sigmoid(n_x)
        o_x = n_x * x

        # 这样通道数会增加
        x = torch.cat([x, o_x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, num_stroke, in_channels=1,):
        super(UNet, self).__init__()

        self.start_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # EncoderStage
        self.down1 = Encoder_ss(64, 128)
        self.down2 = Encoder_ss(128, 256)
        self.down3 = Encoder_ss(256, 512)
        self.down4 = Encoder_ss(512, 1024)
        self.down5 = Encoder_ss(1024, 2048)

        # DecoderStage
        self.up1 = Decoder(2048, 1024)
        self.up2 = Decoder(1024, 512)
        self.up3 = Decoder(512, 256)
        self.up4 = Decoder(256, 128)
        self.up5 = Decoder(128, 64)

        self.end_conv = nn.Conv2d(64, num_stroke, kernel_size=3*3, padding='same', bias=False)

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.down5(x5)
        # x6_p = self.pool(x6)

        x = self.up1(x5, x)
        x = self.up2(x4, x)
        x = self.up3(x3, x)
        x = self.up4(x2, x)
        x = self.up5(x1, x)

        x = self.end_conv(x)

        return x






class Generator(nn.Module):

    def __init__(self, num_stokes, in_channels):
        super(Generator, self).__init__()
        self.model = UNet(num_stokes, in_channels)

    def forward(self, x,):
        x = self.model(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """

        :param inpyt_nc: 输入通道个数
        :param ndf: 第一次卷积后的通道数
        :param n_layers: 卷积层层数
        """
        super(Discriminator, self).__init__()

        kw = 4
        # 核大小
        padw = 1
        # 填充
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias=False),
                    nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            # 前一次扩大倍数
            nf_mult = min(2**n, 8)
            # 通道数扩大倍数
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)

        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, image):

        return self.model(image)




if __name__ == '__main__':
    image = torch.rand(5, 3, 256, 256)
    # model = UNet(10, 3)
    #
    # back = model(image)
    # print(back.shape)
    d = Discriminator(3)
    back = d(image)
    print(back.shape)

    pass