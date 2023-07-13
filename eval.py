#%%
import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from PIL import ImageDraw
from datasets.Data import DatasetAGE
from models.Losses import *
# from models.unet import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
import pandas as pd

utils.log('start evaluation')

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='mle', type=str, help="get center type")
parser.add_argument('--show', default=1, type=int, help="get center type")
Flags, _ = parser.parse_known_args()
mode = Flags.mode

#%% Dataset
data_dir = os.path.join('/home/liuming', 'MIPAV', 'Datasets', 'AMD')
dataset_test = DatasetAGE(data_dir, label_type='Gau_400', mode='val')
dataloader_test = DataLoader(dataset_test)
utils.log('Load Data successfully')

out_dir = os.path.join(data_dir, 'ValResult')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
utils.checkpath(out_dir)
metrics = LocalizationMetrics()
#==============================================================================
# load model
#==============================================================================
path = "/home/liuming/MIPAV/AMDFoveaLocalization/results/T1105-1357-Result/Model"
modelname = utils.all_files_under(path, '.pkl')
model = torch.load(modelname[-1])
utils.log('Build Model successfully')

df = pd.DataFrame(columns=[
    'AMD_NAME',
    'X_Pred',
    'Y_Pred',
    'X_Truth',
    'Y_Truth',
])
#==============================================================================
# evaluation
#==============================================================================
err = []
with torch.no_grad():
    model.eval()
    pbar = tqdm(dataloader_test, ncols=100)
    pbar.set_description('Test')
    log_Test, temp = {}, {}
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch = batch_data[0]
        y_batch = batch_data[1]
        filename = batch_data[2][0]
        # print(filename)
        if torch.cuda.is_available():
            x_data = x_batch.cuda()
            y_true = y_batch.cuda()
        # forward
        y_pred = model(x_data)["out"]
        y_pred = torch.sigmoid(y_pred)

        pred = y_pred.cpu().data.numpy()
        true = y_true.cpu().data.numpy()

        # type1
        err.append(metrics.EuclideanDistance(true, pred, mode))
        pred = np.squeeze(pred)
        true = np.squeeze(true)
        pred_pos = GetPeak(pred, mode)
        true_pos = GetPeak(true, mode)
        
        poslist = pred_pos + true_pos
        df.loc[n_step] = [filename + '.jpg'] + poslist

        # save
        if Flags.show == 1:
            distmap = utils.array2image(pred / np.max(pred))
            distmap.save(os.path.join(out_dir, filename + '_pred.png'))

            image = np.squeeze(x_data.cpu().data.numpy())
            image = np.transpose(image, (1, 2, 0))
            image = utils.array2image(image)
            draw = ImageDraw.Draw(image)
            draw.text((poslist[0], poslist[1]), '+', fill=(255, 0, 0)) # predict
            draw.text((poslist[2], poslist[3]), '+', fill=(0, 255, 0)) # ground truth
            image.save(os.path.join(out_dir, filename + '.png'))

df.to_csv(os.path.join(out_dir, 'Localization_Results_{}.csv'.format(mode)))

print(np.mean(np.array(err)))