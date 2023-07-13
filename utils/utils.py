import argparse
import datetime
import os
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance


def SearchFreeGPU(interval=60, threshold=0.5):
    while True:
        qargs = ['index', 'memory.free', 'memory.total']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        gpus = pd.DataFrame(np.zeros([len(results), 3]), columns=qargs)
        for i, line in enumerate(results):
            info = line.strip().split(',')
            gpus.loc[i, 'index'] = info[0]
            gpus.loc[i, 'memory.free'] = float(info[1].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'memory.total'] = float(info[2].upper().strip().replace('MIB', ''))
            gpus.loc[i, 'Freerate'] = gpus.loc[i, 'memory.free'] / gpus.loc[i, 'memory.total']

        maxrate = gpus.loc[:, "Freerate"].max()
        index = gpus.loc[:, "Freerate"].idxmax()
        if maxrate > threshold:
            print('GPU index is: {}'.format(index))
            return str(index)
        else:
            print('Searching Free GPU...')
            time.sleep(interval)


def dic2txt(filename, dic):
    fp = open(filename, 'w')
    for key in dic:
        fp.writelines(key + ':\t' + str(dic[key]) + '\n')
    fp.close()
    return True


def MergeLog(logs, temp, n_step):
    for key in temp:
        if key in logs:
            logs[key] = (logs[key] * n_step + temp[key]) / (n_step + 1)
        else:
            logs[key] = (0.0 * n_step + temp[key]) / (n_step + 1)

    return logs


def ShowFlags(FLAGS):
    log('Argparse Settings')
    for i in vars(FLAGS):
        if len(i) < 8:
            print(i + '\t\t------------  ' + str(vars(FLAGS)[i]))
        else:
            print(i + '\t------------  ' + str(vars(FLAGS)[i]))
    print()

    return FLAGS


def checkpath(path):
    try:
        os.makedirs(path)
        # print('creat ' + path)
    except OSError:
        pass


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


def array2image(arr):
    if (np.max(arr) <= 1):
        image = Image.fromarray((arr * 255).astype(np.uint8))
    else:
        image = Image.fromarray((arr).astype(np.uint8))

    return image


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
