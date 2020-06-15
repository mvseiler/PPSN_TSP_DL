#!/usr/bin/env python
# coding: utf-8

import csv
import argparse
import numpy as np
import torch
from utils.transformers import image_tranformer, fold_image_collector

parser = argparse.ArgumentParser(description='Set Folds')
parser.add_argument('folds', type=int, help='Number of Fold to use')
args = parser.parse_args()
print('Folds: ', args.folds)

FOLDS = args.folds
BATCH_SIZE = 8
WIN_SIZE = 512

MODEL_PATH = '../Models/model_1_points_{}'
#MODEL_PATH = '../Models/model_1_mst_{}'
#MODEL_PATH = '../Models/model_1_nng_{}'
#MODEL_PATH = '../Models/model_2_points_mst_{}'
#MODEL_PATH = '../Models/model_2_points_mst_nng_{}'

SOLVER_NAME = 'ann_1_points'
#SOLVER_NAME = 'ann_1_mst'
#SOLVER_NAME = 'ann_1_nng'
#SOLVER_NAME = 'ann_2_points_mst'
#SOLVER_NAME = 'ann_2_points_mst_nng'


IMAGE_DIR = '../instances/evolved1000/Points'
#IMAGE_DIR = '../instances/evolved1000/MST/'
#IMAGE_DIR = '../instances/evolved1000/NNG/'
#IMAGE_DIR = ['../instances/evolved1000/', '../instances/evolved1000/MST_plots/']

FOLDS_PATH = '../instances/evolved1000/sophisticated_1000_folds.csv'


transformer = image_tranformer(resize=WIN_SIZE, gray_scale=True, random_flip=False,
                               random_rotate=False, invert_image=False)

_, test_data_loader, val_paths = fold_image_collector(IMAGE_DIR, FOLDS_PATH, FOLDS, BATCH_SIZE, transformer,
                                                      is_classification=True, shuffle=False,
                                                      also_include_par=True, num_workers=0, include_val_paths=True)


model = torch.load(MODEL_PATH.format(FOLDS)).cuda()


batch_outputs = []
y_trues = []
pqr_scores = []
par_scores = []
for batch in iter(test_data_loader):
    X, y_true, pqr, par = batch #
    pred_y = torch.exp(model(X.cuda())).cpu().detach().numpy() #,
    batch_outputs.append(pred_y)
    y_trues.append(y_true.detach().numpy())
    pqr_scores.append(pqr.detach().numpy())
    par_scores.append(par.detach().numpy())
    del X, y_true, pqr, par

batch_outputs = np.concatenate(batch_outputs)
y_trues = np.concatenate(y_trues)
pqr_scores = np.concatenate(pqr_scores)
par_scores = np.concatenate(par_scores)

tuned_pqr = []
r = 100000
for i in range(r):
    t = i / r
    tuned_pqr.append(np.mean(pqr_scores[np.arange(pqr_scores.shape[0]), (batch_outputs[:,0] <= t) * 1]))
tuned_pqr = np.array(tuned_pqr)

tuned_par = []
r = 100000
for i in range(r):
    t = i / r
    tuned_par.append(np.mean(par_scores[np.arange(par_scores.shape[0]), (batch_outputs[:,0] <= t) * 1]))
tuned_par = np.array(tuned_par)

best_pqr = tuned_pqr.min()
best_par = tuned_par.min()
best_pqr_threshold = np.max(np.where(tuned_pqr == best_pqr)[0]) / r
best_par_threshold = np.max(np.where(tuned_par == best_par)[0]) / r

final_pqr_scores = pqr_scores[np.arange(pqr_scores.shape[0]), (batch_outputs[:,0] <= best_pqr_threshold) * 1]
final_par_scores = par_scores[np.arange(par_scores.shape[0]), (batch_outputs[:,0] <= best_par_threshold) * 1]

# Write CSV File
rows = []
for i, path in enumerate(val_paths):
    if type(path) == list:
        group, prob = path[0].split('/')[-2:]
    else:
        group, prob = path.split('/')[-2:]
    group = 'sophisticated-eax' if group == 'eax---lkh---sophisticated' else 'sophisticated-lkh'
    prob = prob.split('.')[0]
    rows.append({'prob' : prob, 'group' : group, 'measure' : 'PAR10',
                 'solver' : SOLVER_NAME, 'performance' : final_par_scores[i]})
    rows.append({'prob' : prob, 'group' : group, 'measure' : 'PQR10',
                 'solver' : SOLVER_NAME, 'performance' : final_pqr_scores[i]})
print(rows[:20])

with open(MODEL_PATH.replace('models', 'performance_new').format(FOLDS) + '.csv', 'w', newline='') as csvfile:
    fieldnames = ['prob', 'group', 'measure', 'solver', 'performance']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for row in rows:
        writer.writerow(row)
