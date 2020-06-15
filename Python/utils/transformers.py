from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from PIL.ImageOps import invert
import numpy as np
import torch, re
from torch import stack, Tensor, LongTensor, zeros, cat, eye, load
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from PIL import Image
import os

def is_valid_file(path):
    name = path.split('/')[-1]
    ending = name.split('-')[-1]
    return ending == 'simple.png' or ending == 'sophisticated.png'

def image_tranformer(resize=512, gray_scale=True, random_flip=True, random_rotate=True, invert_image=False):
    transformers = []
    rotators = [transforms.Lambda(lambda x: x),
               transforms.Lambda(lambda x: x.rotate(90)),
               transforms.Lambda(lambda x: x.rotate(180)),
               transforms.Lambda(lambda x: x.rotate(270))]
    if gray_scale:
        transformers.append(transforms.Grayscale())
    if resize > 0:
        transformers.append(transforms.Resize(resize))
    if random_flip:
        transformers.append(transforms.RandomHorizontalFlip())
        transformers.append(transforms.RandomVerticalFlip())
    if random_rotate:
        transformers.append(transforms.RandomChoice(rotators))
    if invert_image:
        transformers.append(transforms.Lambda(lambda x: invert(x)))

    transformers.append(transforms.ToTensor())
    return transforms.Compose(transformers)

def fold_image_collector(image_dir, folds_path, folds, batch_size, transformer, features_path=None,
                         is_classification=False, shuffle=True, also_include_par=False, include_val_paths=False, num_workers=4):
    data_pd = pd.read_csv(folds_path)
    keys = data_pd['Path']
    data_pd = data_pd.set_index('Path')
    if features_path is not None:
        data_feat = pd.read_csv(features_path)
        data_feat = data_feat.set_index('path')
    else:
        data_feat = None

    X_train = []
    Feat_train = []
    X_val = []
    Feat_val = []
    y_train = []
    y_val = []
    pqr_train = []
    pqr_val = []
    par_train = []
    par_val = []
    val_paths = []

    for key in keys:
        if data_pd.loc[key]['Fold'] == folds:
            if type(image_dir) is list:
                val_p = [i_dir + key for i_dir in image_dir]
                val_paths.append(val_p)
                im = [Image.open(p.replace('.tsp', '.png')) for p in val_p]
                im = [transformer(i) for i in im]
                X_val.append(torch.cat(im, dim=0))
            else:
                val_p = image_dir + key
                val_paths.append(val_p)
                im = Image.open(val_p.replace('.tsp', '.png'))
                X_val.append(transformer(im))
            if data_feat is not None:
                Feat_val.append(torch.tensor(data_feat.loc[key][1:]))
            y_val.append(torch.tensor([data_pd.loc[key]['EAX.LOG.PQR10'], data_pd.loc[key]['LKH.LOG.PQR10']]))
            pqr_val.append(torch.tensor([data_pd.loc[key]['EAX.PQR10'], data_pd.loc[key]['LKH.PQR10']]))
            par_val.append(torch.tensor([data_pd.loc[key]['EAX.PAR10'], data_pd.loc[key]['LKH.PAR10']]))
        else:
            if type(image_dir) is list:
                im = [Image.open((i_dir + key).replace('.tsp', '.png')) for i_dir in image_dir]
                im = [transformer(i) for i in im]
                X_train.append(torch.cat(im, dim=0))
            else:
                im = Image.open((image_dir + key).replace('.tsp', '.png'))
                X_train.append(transformer(im))
            if data_feat is not None:
                Feat_train.append(torch.tensor(data_feat.loc[key][1:]))
            y_train.append(torch.tensor([data_pd.loc[key]['EAX.LOG.PQR10'], data_pd.loc[key]['LKH.LOG.PQR10']]))
            pqr_train.append(torch.tensor([data_pd.loc[key]['EAX.PQR10'], data_pd.loc[key]['LKH.PQR10']]))
            par_train.append(torch.tensor([data_pd.loc[key]['EAX.PAR10'], data_pd.loc[key]['LKH.PAR10']]))

    X_train = torch.stack(X_train, dim=0)
    X_val = torch.stack(X_val, dim=0)
    y_train = torch.stack(y_train, dim=0)
    y_val = torch.stack(y_val, dim=0)
    pqr_train = torch.stack(pqr_train, dim=0)
    pqr_val = torch.stack(pqr_val, dim=0)
    par_train = torch.stack(par_train, dim=0)
    par_val = torch.stack(par_val, dim=0)
    if features_path is not None:
        Feat_train = torch.stack(Feat_train, dim=0)
        Feat_val = torch.stack(Feat_val, dim=0)
        Feat_train = (Feat_train - torch.mean(Feat_train, dim=0)) / torch.std(Feat_train, dim=0)
        Feat_val = (Feat_val - torch.mean(Feat_val, dim=0)) / torch.std(Feat_val, dim=0)

    if is_classification:
        y_train = torch.argmin(y_train, dim=-1)
        y_val = torch.argmin(y_val, dim=-1)

    if also_include_par and features_path is not None:
        train_set =  TensorDataset(X_train, Feat_train, y_train, pqr_train, par_train)
        val_set = TensorDataset(X_val, Feat_val, y_val, pqr_val, par_val)

    elif also_include_par and features_path is None:
        train_set =  TensorDataset(X_train, y_train, pqr_train, par_train)
        val_set = TensorDataset(X_val, y_val, pqr_val, par_val)

    elif not also_include_par and features_path is not None:
        train_set =  TensorDataset(X_train, Feat_train, y_train, pqr_train)
        val_set = TensorDataset(X_val, Feat_val, y_val, pqr_val)

    else:
        train_set =  TensorDataset(X_train, y_train, pqr_train)
        val_set = TensorDataset(X_val, y_val, pqr_val)

    train_ds = DataLoader(train_set, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    val_ds = DataLoader(val_set, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
    if include_val_paths:
        return (train_ds, val_ds, val_paths)
    else:
        return (train_ds, val_ds)
