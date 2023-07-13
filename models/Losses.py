import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class WeightedMSE(nn.Module):
    def __init__(self, threshold=0.3, amp=3):
        super(WeightedMSE, self).__init__()
        self.th = threshold
        self.amp = amp * 1.0

    def forward(self, y_true, y_pred):
        region = (y_true > self.th).float()
        loss = (1 + self.amp * region) * torch.pow((y_true - y_pred), 2)
        loss = loss.float().mean()

        return loss


class PositionLoss(nn.Module):
    def __init__(self, h=512, w=512):
        super(PositionLoss, self).__init__()
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        self.xx = torch.tensor(xx[np.newaxis, :, :]).float().cuda()
        self.yy = torch.tensor(yy[np.newaxis, :, :]).float().cuda()
        self.xx = self.xx.unsqueeze(0)
        self.yy = self.yy.unsqueeze(0)

    def forward(self, pos_true, y_pred):
        N = pos_true.shape[0]
        xx = self.xx.repeat(N, 1, 1, 1)
        yy = self.yy.repeat(N, 1, 1, 1)

        pos_pred = Variable(torch.zeros(N, 2), requires_grad=True).float().cuda()
        pos_pred[:, 0] = torch.sum(y_pred * xx, [1, 2, 3]) * torch.pow(torch.sum(y_pred, [1, 2, 3]), -1)
        pos_pred[:, 1] = torch.sum(y_pred * yy, [1, 2, 3]) * torch.pow(torch.sum(y_pred, [1, 2, 3]), -1)

        loss_pos = F.mse_loss(pos_pred, pos_true)

        return loss_pos


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list): alpha = torch.Tensor(alpha)
        self.alpha = alpha.cuda()
        self.size_average = size_average

    def forward(self, y_true, y_pred):
        pos = -y_true * ((1.0 - y_pred)**self.gamma) * torch.log(y_pred)
        neg = -(1.0 - y_true) * (y_pred**self.gamma) * torch.log(1.0 - y_pred)
        loss = self.alpha[0] * pos + self.alpha[1] * neg

        if self.size_average:
            loss_focal = loss.mean()
        else:
            loss_focal = loss.sum()

        return loss_focal


class MaskFisherLoss(nn.Module):
    def __init__(self, smooth=0.1):
        super(MaskFisherLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, tensor, mask):
        y_true = y_true + mask
        mask0 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)
        cov0 = 1.0 / vec0.shape[-1] * torch.mm(vec0 - miu0, torch.transpose(vec0 - miu0, 0, 1))

        mask1 = (y_true == 2).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)
        cov1 = 1.0 / vec0.shape[-1] * torch.mm(vec1 - miu1, torch.transpose(vec1 - miu1, 0, 1))

        Sw = torch.det(cov0) + torch.det(cov1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss


class FisherLoss2(nn.Module):
    def __init__(self, smooth=0.1):
        super(FisherLoss2, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, tensor):
        mask0 = (y_true == 0).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)
        cov0 = 1.0 / vec0.shape[-1] * torch.mm(vec0 - miu0, torch.transpose(vec0 - miu0, 0, 1))

        mask1 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)
        cov1 = 1.0 / vec0.shape[-1] * torch.mm(vec1 - miu1, torch.transpose(vec1 - miu1, 0, 1))

        Sw = torch.det(cov0) + torch.det(cov1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss


class BCELoss(nn.Module):
    def __init__(self, weight=None):
        super(BCELoss, self).__init__()
        self.weight = weight

    def forward(self, y_true, y_pred):
        bce = F.binary_cross_entropy(y_pred, y_true, weight=self.weight)

        return bce


class DiceLoss(nn.Module):
    def __init__(self, smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        intersect = (y_pred * y_true).sum()
        union = torch.sum(y_pred) + torch.sum(y_true)
        Dice = (2 * intersect + self.smooth) / (union + self.smooth)
        dice_loss = 1 - Dice

        return dice_loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_true, y_pred):
        mse = F.mse_loss(y_pred, y_true)

        return mse


"""
# 第一版fisherloss，原理上有错误
class FisherLoss(nn.Module):
    def __init__(self, smooth=1):
        super(FisherLoss, self).__init__()
        self.smooth = smooth

    def forward(self, tensor, y_true):
        mask0 = (y_true == 0).repeat(1, tensor.shape[1], 1, 1)
        vec0 = torch.masked_select(tensor, mask0)
        vec0 = torch.reshape(vec0, (tensor.shape[0], tensor.shape[1], -1))
        miu0 = torch.mean(vec0, dim=-1, keepdim=True)
        var0 = torch.var(vec0, dim=-1, keepdim=True)

        mask1 = (y_true == 1).repeat(1, tensor.shape[1], 1, 1)
        vec1 = torch.masked_select(tensor, mask1)
        vec1 = torch.reshape(vec1, (tensor.shape[0], tensor.shape[1], -1))
        miu1 = torch.mean(vec1, dim=-1, keepdim=True)
        var1 = torch.var(vec1, dim=-1, keepdim=True)

        Sw = torch.sum(var0 + var1)  # 类内离散度
        Sb = torch.dist(miu0, miu1)  # 类间距离
        fisherloss = (Sw + self.smooth) / (Sb + self.smooth)

        return fisherloss
"""