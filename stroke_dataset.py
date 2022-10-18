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
        all_img_y_path = [os.path.join(img_y_path, stroke) for stroke in os.listdir(img_y_path)]
        y = list()
        for i in range(len(all_img_y_path)):
            t = cv2.imdecode(np.fromfile(all_img_y_path[i], dtype=np.uint8), -1)
            y.append(t)

        y = np.array(y)

        # print(x.__class__)
        # print(y.__class__)

        x = self.transform(x)
        y = self.transform(y)



        return x, y

    def __len__(self):
        return len(self.x_path)


if __name__ == '__main__':
    root = 'E:/PythonPPP/pythonTest/test'
    img = stroke_Dataset(root)
    trainDataLoader = DataLoader(img, batch_size=1,
                                 shuffle=False, num_workers=1)
    it = trainDataLoader.__iter__()
    img, stroke = it.next()
    print(img.shape)
    print(stroke.shape)
