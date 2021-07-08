# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
import os
import os.path as op
from torch.utils.data import TensorDataset, DataLoader
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
import torch.onnx
from src.utils.AgentSummary import SummaryLogger
from src.data_utils.process_data import *
from src.utils.build_models import *
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from src.utils.pytorchtools import EarlyStopping
import shutil
import sys
from augmentation import *
import tsaug as ts
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run_path', default='./', help=' ')
parser.add_argument('--n_epochs', type=int, default=1500, help=' ')
parser.add_argument('--n_iters', type=int, default=5, help=' ')
parser.add_argument('--datasets', default='all', help=' ')
parser.add_argument('--augment', choices=['baseline', 'rand_augment', 'w_augment', 'atrim_augment'], help=' ')
parser.add_argument('--param_M', type=float, default=10, help=' ')
args = parser.parse_args()

da_methods_mapping = { 
    'jitter':           { 'func' : jitter,          'params' : { 'sigma' : [0.01, 0.5] }},
    'timewarp':         { 'func' : time_warp,       'params' : { 'sigma': [0.01, 0.5], 'knot': [3, 5]}},
    'window_slice':     { 'func' : window_slice,    'params' : { 'reduce_ratio' : [0.95, 0.6] }},
    'window_warp':      { 'func' : window_warp,     'params' : { 'window_ratio': [0.1, 0.9], 'scales_max': [0.1, 5]}},
    'scaling':          { 'func' : scaling,         'params' : { 'sigma' : [0.1, 2.] }},
    'magnitude_warp':   { 'func' : magnitude_warp,  'params' : { 'sigma': [0.1, 2.], 'knot': [3, 5]}},
    'permutation':      { 'func' : permutation,     'params' : { 'max_segments' : [3., 6.] }},
    'dropout':          { 'func' : ts.Dropout,      'params' : { 'p' : [0.05, 0.5] }}
}

def m_to_param(m, m_min, m_max, a_min, a_max):
    dm = m_max - m_min
    da = a_max - a_min
    return a_max + (m - m_max) * (da / dm)

def m_to_method_params(m, method):
    ret = {}
    for param in da_methods_mapping[method]['params']:
        p_min, p_max = da_methods_mapping[method]['params'][param]
        p_val = m_to_param(m, 1.0, 20.0, p_min, p_max) #m_min=1.0, m_max=30.0
        ret[param] = p_val
    return ret

param_dic = { method : m_to_method_params(args.param_M, method) for method in da_methods_mapping }



def select_augmented_sample(inp, label):
    x = inp.numpy()
    y = label.numpy()
    transforms = ['None', 'jitter', 'timewarp',  'window_slice', 'window_warp']
    sampled_op = np.random.choice(transforms, 1)
    if sampled_op[0] == 'None':
        return x
    else:
        x_r = x.reshape((-1, x.shape[1], 1))
        x_r_aug = da_methods_mapping[sampled_op[0]]['func'](x_r, **param_dic[sampled_op[0]])
        return x_r_aug.reshape(-1, x.shape[1])
        
def augment_single_sample(x, y):
    augmented_input = []
    augmented_input.append(x)
    for key in da_methods_mapping:
        if key == 'dropout':
            x_r_aug = da_methods_mapping[key]['func'](fill=0., **param_dic[key]).augment(x)
            augmented_input.append(x_r_aug)
        else:
            x_r = x.reshape((-1, x.shape[1], 1))
            x_r_aug = da_methods_mapping[key]['func'](x_r, **param_dic[key])
            augmented_input.append(x_r_aug.reshape(-1, x.shape[1]))
    return np.concatenate(augmented_input)


def evaluate_augment(inp, label, model):
    x = inp.numpy()
    y = label.numpy()
    augm_len = len(da_methods_mapping.keys())+1
    net_inp = augment_single_sample(x, y)
    label = label.view(-1).long()
    lab = torch.cat(augm_len*[label])
    out = model(torch.from_numpy(net_inp).to(device).float())
    return out, lab

def alpha_trim(inp, label, model):
    x = inp.numpy()
    y = label.numpy()
    len_inp = len(x)
    len_timeseries = x.shape[1]
    augm_len = len(da_methods_mapping.keys())+1
    net_inp = augment_single_sample(x, y)
    label = label.view(-1).long()
    lab = torch.cat(augm_len*[label])
    out = model(torch.from_numpy(net_inp).to(device).float())
    number_of_bins = np.arange(0,len(da_methods_mapping.keys())+2)
    single_loss = nn.CrossEntropyLoss(reduction='none')
    rloss = single_loss(out, lab.to(device)).reshape(augm_len, -1)
    rloss = torch.transpose(rloss, 0,1)
    rinput = np.transpose(net_inp.reshape(augm_len, len_inp, len_timeseries), (1,0,2))
    sval, spos = rloss.sort()
    new_pos = spos[:, 1:-1]
    sol = np.concatenate(rinput[np.indices(new_pos.shape)[0], new_pos.cpu().numpy()])
    rlab = np.transpose(lab.reshape(augm_len, len_inp), (1,0))
    new_lab = np.concatenate(rlab.cpu().numpy()[np.indices(new_pos.shape)[0], new_pos.cpu().numpy()])
    mH, _ = np.histogram(np.concatenate(new_pos.cpu().numpy()), bins=number_of_bins)
    return sol, torch.from_numpy(new_lab).long(), mH


def alpha_trim_epoch_trainer(model, train_loader, optimizer, criterion, logger):
    model.train()
    _losses, pred_list, label_list, myhistlist = [], [], [], []
    m = nn.Softmax(dim=-1)
    for X, label in train_loader:
        model.zero_grad()
        with torch.set_grad_enabled(False):
            aug_x, label, hist_ = alpha_trim(X, label, model)
            myhistlist.append(hist_)
        out = model(torch.from_numpy(aug_x).to(device).float())
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        loss.backward()
        optimizer.step()
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label.cpu().detach().numpy().reshape(-1))
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    acc = accuracy_score(y_true, y_pred)
    train_loss = np.average(_losses)
    logger.add_scalar('train_loss', train_loss)
    logger.add_scalar('train_acc', acc)
    hist_sum = np.sum(myhistlist,axis=0)
    logger.add_vector('histos', hist_sum)
    return train_loss, acc

def trainable_weight_augment_epoch_trainer(model, train_loader, optimizer, criterion, logger):
    model.train()
    _losses, pred_list, label_list = [], [], []
    m = nn.Softmax(dim=-1)
    # counter = 0
    for X, label in train_loader:
        model.zero_grad()
        optimizer.zero_grad()
        out, lab = evaluate_augment(X, label, model)
        loss, norm_vector = criterion(out, lab.to(device))   
        _losses.append(loss.item())
        loss.backward()
        optimizer.step()
        logger.add_vector('weight_vector', norm_vector.cpu().detach().numpy())
        # counter += 1
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(lab.cpu().detach().numpy().reshape(-1))
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    acc = accuracy_score(y_true, y_pred)
    train_loss = np.average(_losses)
    logger.add_scalar('train_loss', train_loss)
    logger.add_scalar('train_acc', acc)
    return train_loss, acc

def rand_augment_epoch_trainer(model, train_loader, optimizer, criterion, logger):
    model.train()
    _losses, pred_list, label_list = [], [], []
    m = nn.Softmax(dim=-1)
    for X, label in train_loader:
        model.zero_grad()
        with torch.set_grad_enabled(False):
            aug_x = select_augmented_sample(X, label)
        out = model(torch.from_numpy(aug_x).to(device).float())
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        loss.backward()
        optimizer.step()
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label.cpu().detach().numpy().reshape(-1))
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    acc = accuracy_score(y_true, y_pred)
    train_loss = np.average(_losses)
    logger.add_scalar('train_loss', train_loss)
    logger.add_scalar('train_acc', acc)
    return train_loss, acc

def baseline_epoch_trainer(model, train_loader, optimizer, criterion, logger):
    model.train()
    _losses, pred_list, label_list = [], [], []
    m = nn.Softmax(dim=-1)
    for X, label in train_loader:
        model.zero_grad()
        out = model(X.float().to(device))
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        loss.backward()
        optimizer.step()
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label.cpu().detach().numpy().reshape(-1))
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    acc = accuracy_score(y_true, y_pred)
    train_loss = np.average(_losses)
    logger.add_scalar('train_loss', train_loss)
    logger.add_scalar('train_acc', acc)
    return train_loss, acc


class wLoss(nn.Module):
    def __init__(self, number_of_aug):
        super(wLoss,self).__init__()
        self.number_of_aug = number_of_aug
        self.weight_vector = nn.Parameter(torch.Tensor(number_of_aug*[1/number_of_aug]))
    def forward(self,x,label):
        single_loss = nn.CrossEntropyLoss(reduction='none')
        rloss = single_loss(x, label.to(device)).reshape(self.number_of_aug, -1)
        rloss = torch.transpose(rloss, 0,1)
        m = nn.Softmax(dim=-1)
        normalized_weight_vector = m(self.weight_vector)
        weighted_loss = torch.matmul(rloss, normalized_weight_vector.to(device))
        loss = weighted_loss.mean()
        return loss, normalized_weight_vector

def evaluate_model(model, test_loader, path):
    pred_list, label_list, out_list = [], [], []
    model.eval()
    m = nn.Softmax(dim=-1)
    for x, label in test_loader:
        out = model(x.float().to(device))
        out = m(out)
        _, pred = torch.max(out, 1)
        out_list.append(out.cpu().detach().numpy())
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label)
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    full_output = np.concatenate(out_list)
    y_hat_df = pd.DataFrame(y_true, columns=['y_true']).join(pd.DataFrame(y_pred, columns=['y_pred']))
    y_hat_df.to_csv(op.join(path, 'df_eval.csv'))

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics = {}
    metrics['acc'] = acc
    metrics['b_acc'] = b_acc
    metrics['prec'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.to_csv(op.join(path, 'metrics.csv'))
    return full_output


def evaluate_model_with_validation(model, test_loader, path):
    pred_list, label_list, out_list = [], [], []
    model.eval()
    m = nn.Softmax(dim=-1)
    for x, label in test_loader:
        out = model(x.float().to(device))
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label)
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)

    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    metrics = {}
    metrics['acc'] = acc
    metrics['b_acc'] = b_acc
    metrics['prec'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.to_csv(op.join(path, 'metrics_validation.csv'))


def epoch_validation(model, valid_loader, logger):
    pred_list, label_list, _losses = [], [], []
    model.eval()
    m = nn.Softmax(dim=-1)
    criterion = nn.CrossEntropyLoss()
    for x, label in valid_loader:
        out = model(x.float().to(device))
        label = label.view(-1).long()
        loss = criterion(out, label.to(device))
        _losses.append(loss.item())
        out = m(out)
        _, pred = torch.max(out, 1)
        pred_list.append(pred.cpu().detach().numpy().reshape(-1))
        label_list.append(label)
    y_pred = np.concatenate(pred_list)
    y_true = np.concatenate(label_list)
    valid_loss = np.average(_losses)
    # scheduler.step(valid_loss)
    acc = accuracy_score(y_true, y_pred)
    logger.add_scalar('valid_loss', valid_loss)
    logger.add_scalar('valid_acc', acc)
    return valid_loss, acc



def train_eval_single_model(model, train_loader, valid_loader, test_loader, n_epochs, path, augment, class_weights, validate):
    logger = SummaryLogger(path)

    if augment == 'baseline':
        epoch_trainer = baseline_epoch_trainer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif augment == 'rand_augment':
        epoch_trainer = rand_augment_epoch_trainer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    elif augment == 'w_augment':
        epoch_trainer = trainable_weight_augment_epoch_trainer
        augm_len = len(da_methods_mapping.keys())+1
        criterion = wLoss(augm_len)
        params = list(criterion.parameters())
        params += list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=0.001)
    elif augment == 'atrim_augment':
        epoch_trainer = alpha_trim_epoch_trainer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        sys.exit('invalid augment method')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=0.0001)
    current_valid_loss = 100000
    model_file_name = op.join(path, 'checkpoint.pt')
    early_stopping = EarlyStopping(patience=150, verbose=True, path=path)
    for epoch in range(n_epochs):
        counter = 0
        loss, acc = epoch_trainer(model, train_loader, optimizer, criterion, logger)
        if validate == True:
            valid_loss, valid_acc = epoch_validation(model, valid_loader, logger)
            loss_for_validation = valid_loss
            print(epoch, loss, acc, valid_loss, valid_acc)
        else:
            loss_for_validation = loss
            print(epoch, loss, acc)
        scheduler.step(loss_for_validation)
        logger.add_scalar('learn_rate', scheduler._last_lr)
        # if loss_for_validation < current_valid_loss:
        #     torch.save(model.state_dict(), model_file_name) 
        #     current_valid_loss = loss_for_validation
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break       
    logger.close()
    final_model_file_name = op.join(path, 'checkpoint_final.pt')
    torch.save(model.state_dict(), final_model_file_name) 
    model.load_state_dict(torch.load(model_file_name))
    evaluate_model_with_validation(model, valid_loader, path)
    net_out = evaluate_model(model, test_loader, path)
    return net_out


def run_experiments(run_path, dataset, n_iters, n_epochs, augment, validate):
    if dataset == 'all':
        list_of_datasets = names_of_datasets
    else:
        list_of_datasets = [dataset]
    data_path = op.join(op.expanduser('~'), 'UCR/')
    full_datasets_metrics = []
    for dataset_name in list_of_datasets:
        print(dataset_name)
        x_train, y_train, x_test, y_test, class_weights = read_UCR_dataset(data_path, dataset_name)
        n_classes = len(np.unique(y_train))
        input_size = x_train.shape[1]
        output_list = []
        _iter = 0
        sss = StratifiedShuffleSplit(n_splits=n_iters, test_size=0.2)
        for train_index, valid_index in sss.split(x_train, y_train):
            valid_loader = build_dataloader(x_train[valid_index], y_train[valid_index], batch_size=32)
            train_loader = build_dataloader(x_train[train_index], y_train[train_index], batch_size=32)
            test_loader = build_dataloader(x_test, y_test, batch_size=64, shuffle=False)
            path = op.join(run_path, 'UCR_results', dataset_name+'_'+str(_iter))
            create_directory(path)
            model = build_model(input_size, n_classes)
            model.to(device)
            net_out = train_eval_single_model(model, train_loader, valid_loader, test_loader, n_epochs, path, augment, class_weights, validate)
            output_list.append(net_out)
            _iter += 1
        average_output = np.mean(output_list, axis=0)            
        y_pred = np.argmax(average_output, 1)       
        acc = accuracy_score(y_test, y_pred)
        b_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        metrics = {}
        metrics['acc'] = acc
        metrics['b_acc'] = b_acc
        metrics['prec'] = prec
        metrics['recall'] = rec
        metrics['f1'] = f1
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
        ensemble_path = op.join(run_path, 'UCR_results', dataset_name+'_ensemble')
        create_directory(ensemble_path)
        df_metrics.to_csv(op.join(ensemble_path, 'metrics.csv'))
        full_datasets_metrics.append(df_metrics)
    summary_results = pd.concat(full_datasets_metrics, axis=1)
    summary_results.columns = list_of_datasets
    dest_path = op.join(run_path, 'summary_results')
    create_directory(dest_path)
    summary_results.T.to_csv(op.join(dest_path, 'full_summaries.csv'))


def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass

is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    # torch.cuda.set_device(args.gpu_number)
    print(torch.cuda.current_device())
else:
    device = torch.device("cpu")

names_of_datasets = names_of_datasets = ['ECG5000', 'EthanolLevel', 'ProximalPhalanxOutlineCorrect', 'MiddlePhalanxOutlineCorrect', 'DistalPhalanxOutlineCorrect', 'Strawberry', 'MixedShapesSmallTrain', 'InlineSkate', 'ECG200', 'ACSF1', \
'Ham', 'Haptics', 'Fish', 'WormsTwoClass', 'Worms']


import time 
date_ = time.strftime("%Y-%m-%d_%H%M")
create_directory(op.join(args.run_path, 'scripts'))
copied_script_name = op.basename(__file__)
print(copied_script_name)
shutil.copy(__file__, op.join(args.run_path, 'scripts', copied_script_name+'_'+date_)) 

run_experiments(args.run_path, args.datasets, args.n_iters, args.n_epochs, args.augment, validate=True)










