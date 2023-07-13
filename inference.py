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
from models.unet import *
from utils import utils
from utils.logger import Logger
from utils.Metrics import *
import pandas as pd

utils.log('start evaluation')

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='region', type=str, help="get center type")
parser.add_argument('--show', default=1, type=int, help="get center type")
Flags, _ = parser.parse_known_args()
mode = Flags.mode

#%% Dataset
data_dir = os.path.join('/home/liuming', 'MIPAV', 'Datasets', 'AMD')
dataset_test = DatasetAGE(data_dir, mode='test')
dataloader_test = DataLoader(dataset_test)
utils.log('Load Data successfully')

out_dir = os.path.join(data_dir, 'TestResult')
temp_dir = os.path.join(data_dir, 'TempResult')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
utils.checkpath(out_dir)
utils.checkpath(temp_dir)
metrics = LocalizationMetrics()
#==============================================================================
# load model
#==============================================================================
path = "/home/liuming/MIPAV/AMDFoveaLocalization/results/T1029-2253-Result/Model"
modelname = utils.all_files_under(path, '.pkl')
model = torch.load(modelname[-1])
utils.log('Build Model successfully')

df_max = pd.DataFrame(columns=['AMD_NAME', 'Fovea_X', 'Fovea_Y'])
df_mle = pd.DataFrame(columns=['AMD_NAME', 'Fovea_X', 'Fovea_Y'])
df_region = pd.DataFrame(columns=['AMD_NAME', 'Fovea_X', 'Fovea_Y'])
#==============================================================================
# evaluation
#==============================================================================
err = []
with torch.no_grad():
    model.eval()
    pbar = tqdm(dataloader_test, ncols=70)
    pbar.set_description('Test')
    log_Test, temp = {}, {}
    for n_step, batch_data in enumerate(pbar):
        # get data
        x_batch = batch_data[0]
        y_batch = batch_data[1]
        filename = batch_data[2][0]
        #print(filename)
        if torch.cuda.is_available():
            x_data = x_batch.cuda()
            y_true = y_batch.cuda()
        # forward
        y_pred = model(x_data)["main_out"]
        y_pred = torch.sigmoid(y_pred)

        pred = np.squeeze(y_pred.cpu().data.numpy())
        poslist = GetPeak(pred, mode='max')
        df_max.loc[n_step] = [filename + '.jpg'] + poslist
        poslist = GetPeak(pred, mode='mle')
        df_mle.loc[n_step] = [filename + '.jpg'] + poslist
        poslist = GetPeak(pred, mode='region')
        df_region.loc[n_step] = [filename + '.jpg'] + poslist
        # print(poslist)
        # save
        if Flags.show == 1:
            distmap = utils.array2image(pred / np.max(pred))
            distmap.save(os.path.join(out_dir, filename + '_pred.png'))

            image = np.squeeze(x_data.cpu().data.numpy())
            image = np.transpose(image, (1, 2, 0))
            image = utils.array2image(image)
            draw = ImageDraw.Draw(image)
            draw.text((poslist[0], poslist[1]), '+', fill=(255, 0, 0))
            # draw.text((poslist[2], poslist[3]), '+', fill=(255, 0, 0))
            image.save(os.path.join(out_dir, filename + '.png'))

df_max.to_csv(os.path.join(out_dir, 'Localization_Results.csv'))
df_max.to_csv(os.path.join(temp_dir, 'Localization_Results_{}_{}.csv'.format('max', modelname[0][-10:-6])))
df_mle.to_csv(os.path.join(temp_dir, 'Localization_Results_{}_{}.csv'.format('mle', modelname[0][-10:-6])))
df_region.to_csv(os.path.join(temp_dir, 'Localization_Results_{}_{}.csv'.format('region', modelname[0][-10:-6])))
