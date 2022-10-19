#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/9/22
# @file ss_gan.py
import time

import torch
import torch.nn as nn
from stroke_dataset import stroke_Dataset


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
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
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

        self.end_conv = nn.Conv2d(64, num_stroke, kernel_size=3, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # print(x.dtype)
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
        # x = self.sigmoid(x)

        return x


class Generator(nn.Module):

    def __init__(self, num_stokes, in_channels):
        super(Generator, self).__init__()
        self.model = UNet(num_stokes, in_channels)

    def forward(self, x,):
        x = self.model(x)
        return x


class Discriminator(nn.Module):
    # 每个通道对应一个二值壁画图像。鉴别器输出维数与特征映射张量的输出维数相同

    def __init__(self, input_nc, output_nc, ndf=64, n_layers=3):
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
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, image):

        return self.model(image)




device = 'cuda' if torch.cuda.is_available() else 'cpu'

generator = Generator(20, 1).to(device)
discriminator = Discriminator(20, 20).to(device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5, 0.999), weight_decay=0.0001)
# 优化器

adv_loss = nn.BCELoss().to(device)
l1_loss = nn.L1Loss().to(device)
la = 10
# la = la.to(device)
# 损失函数

root = 'E:/PythonPPP/pythonTest/test'
dataset = stroke_Dataset(root)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# labels_one = torch.ones(1, 20, 30, 30)

num_epoch = 1000

torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epoch):

    for i, mini_batch in enumerate(dataloader):

        epoch_start_time = time.time()
        images, strokes = mini_batch
        # 这里读进来的数据类型是 torch.uint8

        images = images.float()
        strokes = strokes.float()

        # print(images.dtype, strokes.dtype)

        images = images.to(device)
        strokes = strokes.to(device)

        b, c, h, w = strokes.shape
        # strokes 通道数可能和 生成器生成的通道数不符， 只需要生成器的前 c个通道即可

        labels_one =torch.tensor(1)
        labels_one = labels_one.to(device)

        pred_strokes = generator(images)
        # print(pred_strokes)



        d_strokes = discriminator(pred_strokes)

        labels_one = labels_one.expand_as(d_strokes).float()
        # patchgan d 生成是一个 矩阵， 标签应该也是个矩阵, 这里竟然默认是long型的变量
        # print(labels_one.dtype)



        g_optimizer.zero_grad()

        l_adv = adv_loss(d_strokes[:, :c, :, :].data, labels_one[:, :c, :, :].data)
        # 只算有笔画的通道
        l_l1 = l1_loss(pred_strokes[:, :c, :, :].data, strokes.data)
        g_l = l_adv.data + la * l_l1
        g_l.requires_grad_(True)
        g_l.backward(retain_graph=True)
        g_optimizer.step()

        d_optimizer.zero_grad()
        # 那个变量改变了？？？
        d_l = -l_adv
        d_l.requires_grad_(True)
        # 一直报错啊。。。。。。。。。。
        d_l.backward(retain_graph=True)
        d_optimizer.step()

        if epoch % 50 == 0:
            print(f"epoch: {epoch}, g_l: {g_l}, d_l: {d_l}, time :{time.time() - epoch_start_time}")
        if epoch % 100 == 0:
            image = pred_strokes[:, :c, :, :].data







# if __name__ == '__main__':
#     image = torch.rand(1, 1, 256, 256)
#     # model = Generator(20, 1)
#     # model = UNet(10, 1)
#     # back = model(image)
#     # print(back.shape)
#     # model = UNet(10, 1)
#     #
#     # back = model(image)
# #     # print(back)
#     d = Discriminator(1, 1)
#     back = d(image)
#     print(back.shape)

#     pass