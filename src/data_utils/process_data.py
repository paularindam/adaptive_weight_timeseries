# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os.path as op
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.utils import class_weight


def read_UCR_dataset(data_path, dataset_name):
    dataset_path = op.join(data_path, dataset_name)
    df_train = pd.read_csv(op.join(dataset_path, dataset_name+'_TRAIN.tsv'), delim_whitespace=True,header=None)
    df_test = pd.read_csv(op.join(dataset_path, dataset_name+'_TEST.tsv'), delim_whitespace=True,header=None)
    x_train = df_train[df_train.columns[1:]].values
    y_train = df_train[0].values
    x_test = df_test[df_test.columns[1:]].values
    y_test = df_test[0].values
    # y_min = np.amin(y_train)
    _, y_train = np.unique(y_train,return_inverse=1)
    _, y_test = np.unique(y_test,return_inverse=1)

    if np.isnan(np.sum(x_train)):
        x_train =  np.nan_to_num(x_train)

    std_ = x_train.std(axis=1, keepdims=True)
    mean_ = x_train.mean(axis=1, keepdims=True)
    # import pdb; pdb.set_trace()
    std_[std_ == 0] = 1.0
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

    std_ = x_test.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    # import pdb; pdb.set_trace()
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    # print(class_weights)

    # missing normalization
    # return x_train, y_train-y_min, x_test, y_test-y_min, class_weights
    return x_train, y_train, x_test, y_test, class_weights
    
def build_dataloader(x_data, y_data, batch_size, shuffle=True):
        train_data = TensorDataset(torch.from_numpy(x_data).float(), torch.from_numpy(y_data))
        train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, drop_last=False)
        return train_loader
