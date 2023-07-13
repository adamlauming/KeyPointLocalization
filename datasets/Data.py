'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2020-10-28 17:13:50
'''
import os
import sys
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets.utils_data as utils


class DatasetAGE(Dataset):
    def __init__(self, data_dir, label_type='Gau_400', mode='train'):
        super().__init__()
        self.mode = mode
        self.label_type = label_type
        self.image_dir = os.path.join(data_dir, 'Images')
        self.test_dir = os.path.join(data_dir, 'TestImages')
        self.label_dir = os.path.join(data_dir, 'Labels_' + label_type)
        self.listfile = os.path.join(data_dir, '{}files.txt'.format(mode))

        self.filenames = utils.txt2list(self.listfile)
        print("Num of {} images:  {}".format(mode, len(self.filenames)))

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        batch_name = self.filenames[index]  # only 1 image name return
        image_arr, label_arr, pos_arr = self.get_img(batch_name, mode=self.mode)

        x_data = self.to_tensor(image_arr.copy()).float()
        y_data = self.to_tensor(label_arr.copy()).float()
        pos_data = torch.tensor(pos_arr.copy()).float()

        return x_data, y_data, batch_name, pos_data

    # load images and labels depend on filenames
    def get_img(self, file_name, mode='train'):
        if mode.startswith('train'):
            image_file = os.path.join(self.image_dir, "{}.bmp".format(file_name))
            label_file = os.path.join(self.label_dir, "{}.png".format(file_name))
            image_im = Image.open(image_file)
            label_im = Image.open(label_file)

            # image_im = utils.random_perturbation(image_im)
            # image_im, label_im = utils.random_geometric(image_im, label_im)
            image_im, label_im = utils.random_transfrom(image_im, label_im)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0
            label_arr = np.array(label_im, dtype=np.float32) / 255.0

            # 灰度图
            # image_arr = image_arr[:, :, np.newaxis]
            # image_arr = np.dstack((image_arr, image_arr, image_arr))

            pos_arr = PosXY3(label_arr)            
            return image_arr, label_arr, pos_arr

        elif mode.startswith('val'):
            image_file = os.path.join(self.image_dir, "{}.bmp".format(file_name))
            label_file = os.path.join(self.label_dir, "{}.png".format(file_name))
            image_im = Image.open(image_file)
            label_im = Image.open(label_file)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0
            label_arr = np.array(label_im, dtype=np.float32) / 255.0
            # image_arr = image_arr[:, :, np.newaxis]
            # image_arr = np.dstack((image_arr, image_arr, image_arr))

            pos_arr = PosXY3(label_arr)

            return image_arr, label_arr, pos_arr

        elif mode.startswith('test'):
            image_file = os.path.join(self.test_dir, "{}.bmp".format(file_name))
            image_im = Image.open(image_file)

            image_arr = np.array(image_im, dtype=np.float32) / 255.0
            # image_arr = image_arr[:, :, np.newaxis]
            # image_arr = np.dstack((image_arr, image_arr, image_arr))

            return image_arr, image_arr, np.array([0, 0])


def PosXY3(mat):
    mat = mat / np.max(mat)
    roi = mat > 0.5
    index = np.argwhere(roi)
    value = mat[roi]
    # print(index)

    y = np.dot(index[:, 0], value) / np.sum(value)
    x = np.dot(index[:, 1], value) / np.sum(value)

    return np.array([x, y])