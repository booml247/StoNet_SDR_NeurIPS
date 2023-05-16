#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 23:09:38 2021

@author: liang257
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import errno
import torch.utils.data
from model import Net, Resize
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from process_data import preprocess_data
from train_DNN import train_DNN
from model import DNN
import argparse
import time
from sklearn.preprocessing import MinMaxScaler



parser = argparse.ArgumentParser(description="Running SM-StoNet")
parser.add_argument('--seed', default=1, type = int, help = 'set seed')
parser.add_argument('--data_path', default="./data/", type=str, help='folder name for loading data')
parser.add_argument('--base_path', default='./result/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='./', type=str, help='folder name for saving model')
parser.add_argument('--data_index', default=0, type=int)
parser.add_argument('--reduce_dim', default=196, type=int)
parser.add_argument('--data_name', default='MNIST', type=str, help='the name of the dataset')
parser.add_argument('--device', default='cpu', type=str, help='the device used to run the experiment')


# Training Setting
parser.add_argument('--num_epochs', default=300, type=int, help='total number of training epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum in SGD')
parser.add_argument('--weight_decay', default=0.000000000001, type=float, help='weight decay in SGD')
parser.add_argument('--batch_size', default=128, type=int, help='batch size for training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for the optimizer')
parser.add_argument('--net_architecture', default=[50, 10], nargs='+', type=int, help='the architecture of the global stonet')
parser.add_argument('--cross_validate_index', default=0, type=int, help='specify which fold of 5 fold cross validation')
parser.add_argument('--regression_flag', default=0, type=bool, help='true for regression and false for classification')

args = parser.parse_args([])

device = args.device
seed = args.seed
data_name = args.data_name
base_path = args.base_path
model_path = args.model_path
cross_validate_index = args.cross_validate_index
regression_flag = args.regression_flag
num_epochs = args.num_epochs
model_path = args.model_path
lr = args.lr
reduce_dim = args.reduce_dim
momentum = args.momentum
weight_decay = args.weight_decay
batch_size = args.batch_size
net_architecture = args.net_architecture
num_hidden = len(net_architecture) - 1
print('reduce dim: ', reduce_dim)


mis_rec = []

#set loss function
if regression_flag:
    loss_func = nn.MSELoss(reduction='sum')
else:
    loss_func = nn.CrossEntropyLoss(reduction='sum')
for cross_validate_index in range(1):
    
    # load data
    x_train, y_train, x_test, y_test = preprocess_data(data_name, cross_validate_index)
    if data_name == 'MNIST':
        x_train, x_test = x_train.view(x_train.size(0), -1), x_test.view(x_test.size(0), -1)
        x_train, x_test, y_train, y_test = x_train[:20000], x_test[:20000], y_train[:20000], y_test[:20000]

    last_hidden_layer_train, last_hidden_layer_test = x_train, x_test
    PATH = base_path + data_name + "/DNN/reduce_dimension_" + str(reduce_dim) + "/cross_" + str(cross_validate_index) + "/"
    
    # scaling
    scaler = MinMaxScaler()
    scaler.fit(last_hidden_layer_train.detach().cpu())
    last_hidden_layer_train = scaler.transform(last_hidden_layer_train.detach().cpu())
    last_hidden_layer_test = scaler.transform(last_hidden_layer_test.detach().cpu())
    
    last_hidden_layer_train = torch.FloatTensor(last_hidden_layer_train).to(device)
    last_hidden_layer_test = torch.FloatTensor(last_hidden_layer_test).to(device)
    
    ntrain = y_train.shape[0]
    ntest = y_test.shape[0]
    
    # train a DNN model using the dimension reduced data
    dnn_net, train_loss_path, test_loss_path, train_accuracy_path, test_accuracy_path = train_DNN(net_architecture, last_hidden_layer_train, last_hidden_layer_test, y_train, y_test, batch_size, lr=lr, momentum=momentum, weight_decay=weight_decay, regression_flag=regression_flag, num_epochs=num_epochs)
    if regression_flag:
        output_test = dnn_net(last_hidden_layer_test)
        test_loss = loss_func(output_test, y_test)
        test_corr = np.corrcoef(y_test.detach().numpy().squeeze(), output_test.detach().numpy().squeeze())[0,1]
        print("test correlation: ", test_corr)
    else:
        with torch.no_grad():
            output_test = dnn_net(last_hidden_layer_test)
            test_loss = loss_func(output_test, y_test)
            prediction_test = output_test.data.max(1)[1]
            test_accuracy = prediction_test.eq(y_test).sum().item() / ntest
            mis_rec.append(1-test_accuracy)
            print('mis_rate: ', 1-test_accuracy)
    
    if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise
    
    import pickle
    filename = PATH + 'loss_path_DNN_dimension_reduce_'+str(net_architecture[-2])+'.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, train_accuracy_path, test_loss_path, test_accuracy_path], f)
    f.close()
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    fig = plt.figure(1)
    plt.plot(train_loss_path, label="train")
    plt.plot(test_loss_path, label="test")
    plt.legend()
    plt.show()
    
    filename = PATH + 'loss_path_DNN_'+str(net_architecture[-1])+'png'
    plt.savefig(filename)
    plt.close()
    
    fig = plt.figure(2)
    plt.plot(train_accuracy_path, label="train")
    plt.plot(test_accuracy_path, label="test")
    plt.legend()
    plt.show()
    
    filename = PATH + 'acc_path_DNN_'+str(net_architecture[-1])+'png'
    plt.savefig(filename)
    plt.close()
    
    import pickle
    filename = PATH + 'result_reduce_DNN_dim_'+str(net_architecture[-2])+'.txt'
    f = open(filename, 'wb')
    pickle.dump([mis_rec], f)
    f.close()
    
    
    
