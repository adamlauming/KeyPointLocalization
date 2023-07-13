import os
import sys
import torch
import pickle
import random
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageEnhance, ImageOps
from skimage import filters
from skimage import measure

# 随机旋转
def random_Rotate(img, label=None):
    rand = int(float(torch.rand(1)-0.5)*60)
    img = img.rotate(rand)
    if label is not None:
        label = label.rotate(rand)
    return img, label

# 随机对比度
def random_Contrast(img):
    v = float(torch.rand(1)) * 2
    if 0.5 <= v <= 1.5:
        return ImageEnhance.Contrast(img).enhance(v)
    else:
        return img

# 随机颜色鲜艳或灰暗
def random_Color(img):
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return ImageEnhance.Color(img).enhance(v)
    else:
        return img

# 随机亮度变换
def random_Brightness(img):  # [0.1,1.9]
    v = float(torch.rand(1)) * 2
    if 0.4 <= v <= 1.5:
        return ImageEnhance.Brightness(img).enhance(v)
    else:
        return img

# 随机变换
def random_transfrom(image, label=None):
    # image = random_Color(image)
    # image, label = random_Rotate(image, label)
    image = random_Contrast(image)
    image = random_Brightness(image)
    # image = random_GaussianBlur(image)
    return image, label

def random_flip(im, flag):
    return ImageOps.mirror(im) if flag else im


def random_rotate(im, degree):
    return im.rotate(degree)

def random_geometric(image, label):
    # 长宽抖动
    # h, w = image.height, image.height
    # h, w = h + 16 * np.random.randint(-3, 4), w + 16 * np.random.randint(-3, 4)
    # image = image.resize((w, h))
    # label = label.resize((w, h))
    # 左右翻转
    flag = random.choice([True, False])
    if flag:
        image = ImageOps.mirror(image)
        label = ImageOps.mirror(label)
    # 旋转
    degree = np.random.randint(-5, 5)
    image = image.rotate(degree)
    label = label.rotate(degree)

    return image, label


def random_perturbation(im):
    gamma = np.exp((np.random.randn(4, 1) / 6))
    # 亮度增强
    en1 = ImageEnhance.Brightness(im)
    im = en1.enhance(gamma[0])
    # 色度增强
    # en2 = ImageEnhance.Color(im)
    # im = en2.enhance(gamma[1])
    # 对比度增强
    en3 = ImageEnhance.Contrast(im)
    im = en3.enhance(gamma[2])
    # 锐度增强
    en4 = ImageEnhance.Sharpness(im)
    im = en4.enhance(gamma[3])

    return im


def z_score(img, *args):
    if not args:
        for idx in range(img.shape[-1]):
            mean = np.mean(img[..., idx])
            std = np.std(img[..., idx])
            img[..., idx] = (img[..., idx] - mean) / std

        return img

    else:
        mask = args[0]
        mask = np.squeeze(mask)
        for idx in range(img.shape[-1]):
            data = img[..., idx]
            data_m = data[mask == 1]
            mean = np.mean(data_m)
            std = np.std(data_m)
            data = (data - mean) / std
            img[..., idx] = data * mask

        return img


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def txt2list(filename):
    """
    read txt content from the txtfile
    """
    names = []
    if (os.path.exists(filename)):
        f = open(filename)
        line = f.readline()
        while line:
            if (line[-1] == '\n'):
                names.append(line[:-1])
            else:
                names.append(line)
            line = f.readline()
        return names  # do not return last \n
    else:
        (_, txtname) = os.path.split(filename)
        log('cannot find {}'.format(txtname))
        sys.exit()


def pad_tensor(tensor, tensor_size):
    org_h, org_w = tensor.shape[1], tensor.shape[2]
    target_h, target_w = tensor_size[0], tensor_size[1]

    if len(tensor.shape) == 4:
        d = tensor.shape[3]
        padded = np.zeros((tensor.shape[0], target_h, target_w, d))
    elif len(tensor.shape) == 3:
        padded = np.zeros((tensor.shape[0], target_h, target_w))

    padded[:, (target_h - org_h) // 2:(target_h - org_h) // 2 + org_h, (target_w - org_w) // 2:(target_w - org_w) // 2 +
           org_w, ...] = tensor

    return padded


def pad_img(img, img_size):
    img_h, img_w = img.shape[0], img.shape[1]
    target_h, target_w = img_size[0], img_size[1]

    if len(img.shape) == 3:
        d = img.shape[2]
        padded = np.zeros((target_h, target_w, d))
    elif len(img.shape) == 2:
        padded = np.zeros((target_h, target_w))

    padded[(target_h - img_h) // 2:(target_h - img_h) // 2 + img_h, (target_w - img_w) // 2:(target_w - img_w) // 2 +
           img_w, ...] = img

    return padded


def crop_tensor(tensor, ori_shape):
    pred_shape = tensor.shape
    assert len(pred_shape) > 2

    if ori_shape == pred_shape:
        return tensor
    else:
        if len(tensor.shape) > 3:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return tensor[:, start_h:end_h, start_w:end_w, :]
        else:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[1], pred_shape[2]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return tensor[:, start_h:end_h, start_w:end_w]


def crop_img(imgs, ori_shape):
    pred_shape = imgs.shape
    assert len(pred_shape) > 1

    if ori_shape == pred_shape:
        return imgs
    else:
        if len(imgs.shape) > 2:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[0], pred_shape[1]
            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[start_h:end_h, start_w:end_w, :]
        else:
            ori_h, ori_w = ori_shape[0], ori_shape[1]
            pred_h, pred_w = pred_shape[0], pred_shape[1]

            start_h, start_w = (pred_h - ori_h) // 2, (pred_w - ori_w) // 2
            end_h, end_w = start_h + ori_h, start_w + ori_w

            return imgs[start_h:end_h, start_w:end_w]


def array2image(arr):
    if (np.max(arr) <= 1):
        image = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        image = Image.fromarray((arr).astype(np.uint8))

    return image


def log(text):
    """
    log status with time label
    """
    print()
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line1 = '=' * 10 + '  ' + nowTime + '  ' + '=' * 10
    length = len(line1)
    leftnum = int((length - 4 - len(text)) / 2)
    rightnum = length - 4 - len(text) - leftnum
    line2 = '*' * leftnum + ' ' * 2 + text + ' ' * 2 + '*' * rightnum
    print(line1)
    print(line2)
    print('=' * len(line1))