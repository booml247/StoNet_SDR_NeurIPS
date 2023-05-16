import torch
import torch.nn as nn
import numpy as np
import os
import errno
from model import DNN
from sklearn.preprocessing import MinMaxScaler


def train_DNN(net_architecture, last_hidden_layer_train, last_hidden_layer_test, y_train, y_test, batch_size, lr, momentum, weight_decay, regression_flag, num_epochs=50):
    
    device = torch.device("cpu")
    

    # build networks
    dim = last_hidden_layer_train.shape[1]
    ntrain = last_hidden_layer_train.shape[0]
    ntest = last_hidden_layer_test.shape[0]
    block_list = []
    block_list.append(nn.Sequential(nn.Linear(dim, net_architecture[0]), nn.Tanh()))
    for i in range(len(net_architecture)-2):
        block_list.append(nn.Sequential(nn.Linear(net_architecture[i], net_architecture[i+1]), nn.Tanh()))
    block_list.append(nn.Linear(net_architecture[-2], net_architecture[-1]))

    num_block = len(block_list)
    num_hidden = num_block - 1

    dnn_net = DNN(block_list)
    # dnn_net = FNN()
    dnn_net.to(device)

    # set optimizer
    # optimizer = torch.optim.Adam(dnn_net.parameters(), lr=lr)
    optimizer = torch.optim.SGD(dnn_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # set loss function
    if regression_flag:
        loss_func = nn.MSELoss()
    else:
        loss_func = nn.CrossEntropyLoss()

    index = np.arange(ntrain)
    train_loss_path = np.zeros(num_epochs)
    test_loss_path = np.zeros(num_epochs)
    train_accuracy_path = np.zeros(num_epochs)
    test_accuracy_path = np.zeros(num_epochs)
    for epoch in range(num_epochs):
        # dnn_net.train()
        epoch_training_loss = 0.0
        total_count = 0

        np.random.shuffle(index)
        for iter_index in range(ntrain // batch_size):
            subsample = index[(iter_index * batch_size):((iter_index + 1) * batch_size)]

            input_data, target = last_hidden_layer_train[subsample,], y_train[subsample]
            input_data, target = input_data.to(device), target.to(device)
            output = dnn_net(input_data)

            loss = loss_func(output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()




        # calculate test set corr
        if regression_flag:
            with torch.no_grad():
                output_train = dnn_net(last_hidden_layer_train)
                train_loss = loss_func(output_train, y_train)
                train_corr = np.corrcoef(y_train.detach().numpy().squeeze(), output_train.detach().numpy().squeeze())[0,1]
                train_loss_path[epoch] = train_loss
                train_accuracy_path[epoch] = train_corr
                print("epoch: ", epoch, ", train loss: ", train_loss, "train correlation: ", train_corr)


                output_test = dnn_net(last_hidden_layer_test)
                test_loss = loss_func(output_test, y_test)
                test_corr = np.corrcoef(y_test.detach().numpy().squeeze(), output_test.detach().numpy().squeeze())[0,1]
                test_loss_path[epoch] = test_loss
                test_accuracy_path[epoch] = test_corr
                print("epoch: ", epoch, ", test loss: ", test_loss, "test correlation: ", test_corr)
        else:
            with torch.no_grad():
                output_train = dnn_net(last_hidden_layer_train)
                train_loss = loss_func(output_train, y_train)
                prediction_train = output_train.data.max(1)[1]
                train_accuracy = prediction_train.eq(y_train).sum().item() / ntrain
                train_loss_path[epoch] = train_loss
                train_accuracy_path[epoch] = train_accuracy
                print("epoch: ", epoch, ", train loss: ", train_loss, "train accuracy: ", train_accuracy)


                output_test = dnn_net(last_hidden_layer_test)
                test_loss = loss_func(output_test, y_test)
                prediction_test = output_test.data.max(1)[1]
                test_accuracy = prediction_test.eq(y_test).sum().item() / ntest
                test_loss_path[epoch] = test_loss
                test_accuracy_path[epoch] = test_accuracy
                print("epoch: ", epoch, ", test loss: ", test_loss, "test accuracy: ", test_accuracy)
    return dnn_net, train_loss_path, test_loss_path, train_accuracy_path, test_accuracy_path