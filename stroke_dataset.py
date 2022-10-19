# 开发者 haotian
# 开发时间: 2022/10/13 16:18
import os
import cv2
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as T
from torch.utils.data import DataLoader


class stroke_Dataset(data.Dataset):
    def __init__(self, root, transform=True, train=True):
        x = os.path.join(root, 'ch_image')
        y = os.path.join(root, 'stroke_image')
        self.x_path = [os.path.join(x, img) for img in os.listdir(x)]
        self.y_path = [os.path.join(y, img.split('.')[0]) for img in os.listdir(x)]

        self.transform = T.Compose([
            # T.Resize((256, 256)),
            torch.tensor,
        ])

    def __getitem__(self, item):
        img_x_path = self.x_path[item]
        img_y_path = self.y_path[item]
        x = cv2.imdecode(np.fromfile(img_x_path, dtype=np.uint8), -1)
        x = self.image_binarization(x)

        x = np.array([x])
        # 增加一维，即通道数
        # x = x[:, :, ::-1].transpose(2, 0, 1)
        # x = x.transpose(2, 0, 1)
        # BGR 转 RGB， HWC 转 CHW
        all_img_y_path = [os.path.join(img_y_path, stroke) for stroke in os.listdir(img_y_path)]
        y = list()
        for i in range(len(all_img_y_path)):
            t = cv2.imdecode(np.fromfile(all_img_y_path[i], dtype=np.uint8), -1)
            # t = t[:, :, ::-1].transpose(2, 0, 1)
            t = self.image_binarization(t)
            # t = t.transpose(2, 0, 1)

            y.append(t)

        y = np.array(y)
        # print(y.shape)

        x = x.copy()
        y = y.copy()

        x = self.transform(x)
        y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.x_path)

    def image_binarization(self, img):
        # 将图片转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 全局阈值二值化， retval，是自动计算出的阈值， dst是返回的图像 ， 130 是阈值， 1是最大值， cv2.是二值化方法
        # 转为灰度图后自动失去通道数
        retval, dst = cv2.threshold(gray, 130, 1, cv2.THRESH_BINARY)
        return dst



if __name__ == '__main__':
    root = 'E:/PythonPPP/pythonTest/test'
    img = stroke_Dataset(root)
    trainDataLoader = DataLoader(img, batch_size=1,
                                 shuffle=False, num_workers=1)
    it = trainDataLoader.__iter__()
    img, stroke = it.next()
    ans: int = 0
    # for i in range(256):
    #     for j in range(256):
    #         if img[0][0][i][j] == 1:
    #             ans += 1
    # print(ans)
    # print(stroke.shape)
