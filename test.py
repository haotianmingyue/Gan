#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2022/9/23
# @file test.py


# l_1 = list()
# # 新的列表
# l_2 = list()
# for i in range(0, len(l_1), 2):
#     l_2.append(l_1[i:i+2])
#
# ----------------------------------------------------损失函数-----------------------------------------


# import numpy as np
# import torch
# import torch.nn as nn
#
# loss_fn = nn.BCELoss()
#
# g_y = torch.rand(1, 20, 30, 30)
# t_y = torch.rand(1, 4, 30, 30)
#
# print(loss_fn(g_y[:, :4, :, :], t_y))


#-------------------------------torch.tensor-------------------
import torch

x = torch.tensor(1)
y = torch.rand(1, 4, 30, 30)
x.expand_as(y)
print(x.shape)