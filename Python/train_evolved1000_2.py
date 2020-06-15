#!/usr/bin/env python
# coding: utf-8

import argparse, os
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.models import MyContextNet
from utils.datagenerators import Model_Trainer
from utils.transformers import image_tranformer, fold_image_collector

parser = argparse.ArgumentParser(description='Set Folds')
parser.add_argument('folds', type=int, help='Number of Fold to use')
args = parser.parse_args()
print('Folds: ', args.folds)
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

EPOCHS = 200
FOLDS = args.folds
BATCH_SIZE = 8
LEARNING_RATE = 0.003
NORM_CLIPPING = 1
WIN_SIZE = 512

IMAGE_DIR = ['../../instances/evolved1000/Points', '../../instances/evolved1000/MST_plots/']
#IMAGE_DIR = ['../../instances/evolved1000/', '../../instances/evolved1000/MST_plots/', '../../instances/evolved1000/NNG/']

FOLDS_PATH = '../../instances/evolved1000/sophisticated_1000_folds.csv'

PATH = '../models/model_2_points_mst_{}'.format(FOLDS)
#PATH = '../models/model_2_points_mst_nng_{}'.format(FOLDS)

transformer = image_tranformer(resize=WIN_SIZE, gray_scale=True, random_flip=True,
                               random_rotate=True, invert_image=False)

train_data_loader, test_data_loader = fold_image_collector(IMAGE_DIR, FOLDS_PATH, FOLDS, BATCH_SIZE, transformer,
                         is_classification=True, shuffle=True, num_workers=0)

model = MyContextNet(input_channels=3, final_channels=2, dropout=0.25)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = OneCycleLR(optimizer, LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=1)
loss_func = nn.NLLLoss() #CrossEntropyLoss()
model = model.cuda()

trainer = Model_Trainer(model, train_data_loader, test_data_loader, optimizer, scheduler,
                        save_path=PATH, norm_clipping=NORM_CLIPPING)

trainer.train(loss_func, EPOCHS)
