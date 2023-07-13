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

from datasets.Data import DatasetAGE
from models.choose_model import seg_model
from models.Losses import *
from utils.Metrics import LocalizationMetrics
from utils import utils
from utils.logger import Logger
from tqdm import tqdm

#%% Settings
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default='40', type=int, help="nums of epoch")
parser.add_argument('--gpu_rate', default=0.5, type=float, help="gpu free rate")
parser.add_argument('--gpu_gap', default=2, type=int, help="check free gpu interval")
parser.add_argument('--batch_size', default=4, type=int, help="batch size")
parser.add_argument('--learning_rate', default=2e-4, type=float, help="initial learning rate for Adam")
parser.add_argument('--val_step', default=5, type=int, help="the frequency of evaluation")
parser.add_argument('--log_cols', default=120, type=int, help="num of columns for log")
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

#%% Dataset
data_dir = os.path.join('/home/liuming', 'MIPAV', 'Datasets', 'AMD')
dataset_train = DatasetAGE(data_dir, label_type='Gau_400', mode='train')
dataloader_train = DataLoader(dataset_train, batch_size=Flags.batch_size, shuffle=True)

dataset_test = DatasetAGE(data_dir, mode='test')
dataloader_test = DataLoader(dataset_test, batch_size=Flags.batch_size, shuffle=True)
utils.log('Load Data successfully')

#%% logger
# logger = Logger(Flags)
# utils.log('Setup Logger successfully')

#%% load model , optimizer and loss
model = seg_model(Flags)
print(torch.cuda.is_available())
model = model.cuda() if torch.cuda.is_available() else model
# summary(model, (1, 640, 640))
optimizer = torch.optim.Adam(model.parameters(), lr=Flags.learning_rate)
criterion = {
    'BCE': BCELoss(),
    'MSE': MSELoss(),
    'WMSE': WeightedMSE(),
}
metrics = LocalizationMetrics()
utils.log('Build Model successfully')

#%% Training
# for epoch in range(Flags.epochs):

############################################################
# Train Period
############################################################
epoch = 0
model.train()
# pbar = tqdm(dataloader_train, ncols=Flags.log_cols)
# pbar.set_description('Epoch {:2d}'.format(epoch))

log_Train, temp = {}, {}
MaxPool = torch.nn.MaxPool2d(2)
for n_step, batch_data in enumerate(dataloader_train):
    # get data
    x_batch = batch_data[0]
    y_batch = batch_data[1]
    mask_batch = batch_data[2]
    # xx = x_batch.cpu().data.numpy()
    # xx = np.transpose(xx, [2, 3, 1, 0])
    # xx = np.squeeze(xx)
    # plt.imshow(xx[..., 0])
    # plt.show()

    if torch.cuda.is_available():
        x_data = x_batch.cuda()
        y_true = y_batch.cuda()
    optimizer.zero_grad()

    # forward
    y_pred = model(x_data)["out"]
    y_pred = torch.sigmoid(y_pred)

    loss = criterion['WMSE'](y_true, y_pred)
    loss.backward()
    optimizer.step()

    y_true = y_true.cpu().data.numpy()
    y_pred = y_pred.cpu().data.numpy()
    dist = metrics.EuclideanDistance(y_true, y_pred)
    print(dist)

    break