#%%
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from datasets.Data import DatasetAGE
from models.Losses import *

from models.choose_model import seg_model
from utils import utils
from utils.logger import Logger
from utils.Metrics import LocalizationMetrics

#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=150, type=int, help="nums of epoch")
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=2, type=int, help="check free gpu interval")
parser.add_argument('--log_cols', default=120, type=int, help="num of columns for log")
parser.add_argument('--workers', default=0, type=int, help="num of workers")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")

parser.add_argument('--batch_size', default=4, type=int, help="batch size")
parser.add_argument('--learning_rate', default=2e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--alpha', default=2, type=float, help="focal loss weigth")
parser.add_argument('--extension', default='bmp', type=str, help="input image extension")
parser.add_argument('--label', default='Gau_400', type=str, help="label type")
parser.add_argument('--datatrain', default='train', type=str, help="select all data or part data train")
parser.add_argument('--savename', default='Result', type=str, help="output folder name")

parser.add_argument('--model', default='FCN', type=str, help="training model")
parser.add_argument('--n_filters', default=64, type=int, help="the filters of one layer")
parser.add_argument('--in_channel', default=3, type=int, help="the channels of input")
parser.add_argument('--n_class', default=1, type=int, help="the numbers of classes -1(background)")
parser.add_argument('--Ulikenet_channel_reduction', default=1, type=int, help="Ulikenet_channel_reduction(the default is Ulikenet_channel/2)")
parser.add_argument('--backbone', default='resnet34', type=str, help="backbone from SegBaseModel")
parser.add_argument('--pretrained', default=False, type=bool, help="whether load pretrained model")
parser.add_argument('--dilated', default=False, type=bool, help="whether dilated")
parser.add_argument('--deep_stem', default=False, type=bool, help="whether deep_stem")
parser.add_argument('--aux', default=False, type=bool, help="whether aux")


Flags, _ = parser.parse_known_args()
utils.ShowFlags(Flags)
os.environ['CUDA_VISIBLE_DEVICES'] = utils.SearchFreeGPU(Flags.gpu_gap, Flags.gpu_rate)

#==============================================================================
# Dataset
#==============================================================================
label_type = Flags.label
data_dir = os.path.join('/home/liuming', 'MIPAV', 'Datasets', 'AMD')
dataset_train = DatasetAGE(data_dir, label_type=label_type, mode=Flags.datatrain)
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

valdir = 'val2' if Flags.datatrain == 'train2' else 'val'
dataset_val = DatasetAGE(data_dir, label_type=label_type, mode=valdir)
dataloader_val = DataLoader(dataset_val, batch_size=Flags.batch_size, num_workers=Flags.workers, shuffle=True)

dataset_test = DatasetAGE(data_dir, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
utils.log('Load Data successfully')

#==============================================================================
# Logger
#==============================================================================
logger = Logger(Flags)
utils.log('Setup Logger successfully')

#==============================================================================
# load model, optimizer, Losses
#==============================================================================
model = seg_model(Flags)
model = model.cuda() if torch.cuda.is_available() else model
# summary(model, (3, 640, 640))
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': BCELoss(),
    'MSE': MSELoss(),
    'FOCAL': FocalLoss(gamma=0, alpha=[Flags.alpha, 1]),
    'PLOSS': PositionLoss(),
}
metrics = LocalizationMetrics()
utils.log('Build Model successfully')

#==============================================================================
# Train model
#==============================================================================
for epoch in range(Flags.epochs + 1):
    ############################################################
    # Train Period
    ############################################################
    model.train()
    pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
    pbar.set_description('Epoch {:2d}'.format(epoch))

    log_Train, temp = {}, {}
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch = batch_data[0]
        y_batch = batch_data[1]
        pos_batch = batch_data[3]
        if torch.cuda.is_available():
            x_data = x_batch.cuda()
            y_true = y_batch.cuda()
            pos_true = pos_batch.cuda()
        optimizer.zero_grad()
        # forward
        y_pred = model(x_data)["out"]
        y_pred = torch.sigmoid(y_pred)
        # backward
        loss_mse = criterion['MSE'](y_true, y_pred)
        # loss_pos = criterion['PLOSS'](pos_true, y_pred)
        loss = loss_mse
        loss.backward()
        optimizer.step()

        # log
        if (epoch % Flags.val_step == 0) or (epoch <= 5):
            y_true = y_true.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            dist_mle = metrics.EuclideanDistance2(y_true, y_pred, 'mle')
            temp['Dist_MLE'] = dist_mle
        temp['Loss'] = loss.item()
        # temp['Focal'] = loss_focal.item()
        # temp['PLOSS'] = loss_pos.item()
        log_Train = utils.MergeLog(log_Train, temp, n_step)
        pbar.set_postfix(log_Train)
    logger.write_tensorboard('1.Train', log_Train, epoch)

    if (not (epoch % Flags.val_step == 0)) and (epoch > 5):
        continue

    ############################################################
    # Test Period
    ############################################################
    print('*' * Flags.log_cols)
    with torch.no_grad():
        model.eval()
        pbar = tqdm(dataloader_val, ncols=Flags.log_cols)
        pbar.set_description('Val')
        log_Test, temp = {}, {}
        for n_step, batch_data in enumerate(pbar):
            # get data
            x_batch = batch_data[0]
            y_batch = batch_data[1]
            pos_batch = batch_data[3]
            if torch.cuda.is_available():
                x_data = x_batch.cuda()
                y_true = y_batch.cuda()
                pos_true = pos_batch.cuda()

            # forward
            y_pred = model(x_data)["out"]
            y_pred = torch.sigmoid(y_pred)

            # loss_focal = criterion['FOCAL'](y_true, y_pred)
            # loss_pos = criterion['PLOSS'](pos_true, y_pred)
            loss_mse = criterion['MSE'](y_true, y_pred)
            loss = loss_mse
            # metric
            y_true = y_true.cpu().data.numpy()
            y_pred = y_pred.cpu().data.numpy()
            dist_max = metrics.EuclideanDistance2(y_true, y_pred, 'max')
            dist_region = metrics.EuclideanDistance2(y_true, y_pred, 'region')
            dist_mle = metrics.EuclideanDistance2(y_true, y_pred, 'mle')
            # log
            temp['Loss'] = loss.item()
            temp['Dist_M'] = dist_max
            temp['Dist_R'] = dist_region
            temp['Dist_MLE'] = dist_mle
            log_Test = utils.MergeLog(log_Test, temp, n_step)
            pbar.set_postfix(log_Test)
        logger.write_tensorboard('2.Val', log_Test, epoch)
        logger.save_model(model, 'Ep{}_Dist_{:.4f}'.format(epoch, log_Test['Dist_MLE']))
    print('*' * Flags.log_cols)
