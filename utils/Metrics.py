'''
Description: 
Author: Ming Liu (lauadam0730@gmail.com)
Date: 2020-10-28 17:13:50
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *
from skimage import filters
import numpy as np
from skimage.measure import regionprops


class LocalizationMetrics(object):
    def __init__(self):
        pass

    def EuclideanDistance(self, y_true, y_pred, mode='max'):
        N = y_true.shape[0]
        dist = np.zeros((N, 2))

        for i in range(N):
            mat_true = y_true[i, 0, :, :]
            mat_pred = y_pred[i, 0, :, :]
            pred_pos = np.array(GetTwoPeak(mat_pred, mode=mode))
            true_pos = np.array(GetTwoPeak(mat_true, mode=mode))
            dist[i, 0] = np.linalg.norm(pred_pos[:2] - true_pos[:2])
            dist[i, 1] = np.linalg.norm(pred_pos[2:] - true_pos[2:])

        return np.mean(dist)

    def EuclideanDistance2(self, y_true, y_pred, mode='max'):
        N = y_true.shape[0]
        dist = np.zeros((N, 1))

        for i in range(N):
            mat_true = y_true[i, 0, :, :]
            mat_pred = y_pred[i, 0, :, :]
            pred_pos = np.array(GetPeak(mat_pred, mode=mode))
            true_pos = np.array(GetPeak(mat_true, mode=mode))
            dist[i, 0] = np.linalg.norm(pred_pos[:2] - true_pos[:2])

        return np.mean(dist)

def GetTwoPeak(mat, mode='max'):
    left = np.ones((mat.shape[0], mat.shape[1] // 2))
    right = np.ones((mat.shape[0], mat.shape[1] - left.shape[1]))
    leftmask = np.hstack((left * 1.0, right * 0))
    rightmask = 1 - leftmask

    if mode == 'max':
        pos_left = PosXY1(mat * leftmask)
        pos_right = PosXY1(mat * rightmask)
    elif mode == 'region':
        pos_left = PosXY2(mat * leftmask)
        pos_right = PosXY2(mat * rightmask)
    elif mode == 'mle':
        pos_left = PosXY3(mat * leftmask)
        pos_right = PosXY3(mat * rightmask)

    return [pos_left[0], pos_left[1], pos_right[0], pos_right[1]]


def GetPeak(mat, mode='max'):
    if mode == 'max':
        pos = PosXY1(mat)
    elif mode == 'region':
        pos = PosXY2(mat)
    elif mode == 'mle':
        pos = PosXY3(mat)

    return [pos[0], pos[1]]



def PosXY1(mat):
    y, x = np.unravel_index(np.argmax(mat), mat.shape)
    return np.array([x, y])


# 修改过 stats = regionprops(mat * 1, coordinates='xy')
def PosXY2(mat):
    mat = mat / np.max(mat)
    mat = mat > 0.5
    # mat1 = np.transpose(mat)
    stats = regionprops(mat * 1)
    center = stats[0].centroid
    x, y = center[1], center[0]

    return np.array([x, y])

def PosXY3(mat):
    mat = mat / np.max(mat)
    roi = mat > 0.5
    index = np.argwhere(roi)
    value = mat[roi]

    y = np.dot(index[:,0], value)/np.sum(value) 
    x = np.dot(index[:,1], value)/np.sum(value) 

    return np.array([x, y])