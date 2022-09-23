#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/9/21
# @file GAN.py

'''
实现生成对抗网络（GAN）
'''

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

image_size = [1, 28, 28]

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(torch.prod(image_size, dtype=torch.int32), 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, torch.prod(image_size, dtype=torch.int32)),
            nn.Tanh(),
        )

    def forward(self, z):
        # shape 0f z : [batchsize, 1*28*28]
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)
        return image


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(torch.prod(image_size, dtype=torch.int32), 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        # shape of images [batchsize, 1, 28, 28]

        prob = self.model(image.reshape(image.shape[0], -1))

        return prob


#training
dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=False,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(28),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=0.5, std=0.5),
                                     ]))




generator = Generator()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.00001)

discriminator = Discriminator()

d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.00001)

# 交叉熵损失
loss_fn = torch.nn.BCELoss()

batch_size = 32
num_epoch = 100
latent_dim = 64

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epoch):
    for i, mini_batch in enumerate(dataloader):
        gt_images, _ = mini_batch
        z = torch.randn(batch_size, latent_dim)
        pred_images = generator(z)

        # 生成器优化
        g_optimizer.zero_grad()
        target = torch.ones(batch_size, 1)
        g_loss = loss_fn(generator(pred_images), target)
        g_loss.backward()
        g_optimizer.step()

        # 判别器优化
        d_optimizer.zero_grad()
        d_loss = 0.5*loss_fn(discriminator(gt_images), torch.ones(batch_size, 1)) + 0.5*loss_fn(discriminator(pred_images.detach()), torch.zeros(batch_size, 1))
        d_loss.backward()
        d_optimizer.step()

        if i % 1000 == 0:
            for index, image in enumerate(pred_images):
                torchvision.utils.save_image(image, f"image_{index}.png")





