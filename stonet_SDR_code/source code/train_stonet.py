import torch
import torch.nn as nn
import numpy as np
import os
import errno
from model import Net

def train_stonet(x_train, x_test, y_train, y_test, PATH, net_index="0", net_architecture=[50, 30, 20], num_epochs=500, subn=128,
                 sigma_list=[0.001, 0.0001, 0.00001], alpha=0.1, MH_step=25,
                 proposal_lr=[0.0000005, 0.00000005, 0.000000005], step_size=0.001, momentum=0.9, weight_decay=0.01,
                 device='cpu', regression_flag=False):
    num_hidden = len(net_architecture) - 1

    # set loss function
    sse = nn.MSELoss(reduction='sum')
    if regression_flag:
        # output_dim = 1
        loss_func = nn.MSELoss()
        loss_func_sum = nn.MSELoss(reduction='sum')
        train_loss_path = np.zeros(num_epochs)
        test_loss_path = np.zeros(num_epochs)
        train_accuracy_path = np.zeros(num_epochs)
        test_accuracy_path = np.zeros(num_epochs)
    else:
        # output_dim = int((y_test.max() + 1).item())
        loss_func = nn.CrossEntropyLoss()
        loss_func_sum = nn.CrossEntropyLoss(reduction='sum')
        train_loss_path = np.zeros(num_epochs)
        test_loss_path = np.zeros(num_epochs)
        train_accuracy_path = np.zeros(num_epochs)
        test_accuracy_path = np.zeros(num_epochs)


    if not os.path.isdir(PATH):
        try:
            os.makedirs(PATH)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                pass
            else:
                raise

    if len(proposal_lr) == 1 and num_hidden > 1:
        temp_proposal_lr = proposal_lr[0]
        proposal_lr = []
        for i in range(num_hidden):
            proposal_lr.append(temp_proposal_lr)

    if len(sigma_list) == 1 and num_hidden > 1:
        temp_sigma_list = sigma_list[0]
        sigma_list = []
        for i in range(num_hidden + 1):
            sigma_list.append(temp_sigma_list)

    # if len(temperature) == 1 and num_hidden > 1:
    #     temp_temperature = temperature[0]
    #     temperature = []
    #     for i in range(num_hidden):
    #         temperature.append(temp_temperature)

    dim = x_train.shape[1]
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    index = np.arange(ntrain)
    sigma_output = sigma_list[-1]


    # build local networks
    block_list = []
    block_list.append(nn.Sequential(nn.Tanh(), nn.Linear(dim, net_architecture[0])))
    for i in range(num_hidden-1):
        block_list.append(nn.Sequential(nn.Tanh(), nn.Linear(net_architecture[i], net_architecture[i+1])))
    block_list.append(nn.Sequential(nn.Tanh(), nn.Linear(net_architecture[num_hidden-1], net_architecture[-1])))

    num_block = len(block_list)

    net = Net(block_list)
    net.to(device)


    # set optimizer
    optimizer_list = []
    for i in range(len(block_list)):
        optimizer_list.append(torch.optim.SGD(net.block_list[i].parameters(), lr=step_size, momentum=momentum,
                                              weight_decay=weight_decay))
        # optimizer_list.append(torch.optim.Adam(net.block_list[i].parameters(), lr=lr))

    for epoch in range(num_epochs):
        np.random.shuffle(index)
        for iter_index in range(ntrain // subn):
            subsample = index[(iter_index * subn):((iter_index + 1) * subn)]

            hidden_list = []
            momentum_list = []
            with torch.no_grad():
                hidden_list.append(net.block_list[0](x_train[subsample,]))
                momentum_list.append(torch.zeros_like(hidden_list[-1]))
                for i in range(1, num_hidden):
                    hidden_list.append(net.block_list[i](hidden_list[-1]))
                    momentum_list.append(torch.zeros_like(hidden_list[-1]))

            foward_hidden = net.block_list[0](x_train[subsample,]).data

            for i in range(hidden_list.__len__()):
                hidden_list[i].requires_grad = True

            for repeat in range(MH_step):
                for layer_index in reversed(range(num_hidden)):
                    # print(layer_index)
                    if hidden_list[layer_index].grad is not None:
                        hidden_list[layer_index].grad.zero_()

                    if layer_index == num_hidden - 1:
                        hidden_likelihood = -loss_func_sum(
                            net.block_list[layer_index + 1](hidden_list[layer_index]),
                            y_train[subsample,]) / sigma_output
                        # print(hidden_likelihood)
                    else:
                        hidden_likelihood = -sse(net.block_list[layer_index + 1](hidden_list[layer_index]),
                                                 hidden_list[layer_index + 1]) / sigma_list[layer_index + 1]
                        # print(hidden_likelihood)
                    if layer_index == 0:
                        hidden_likelihood = hidden_likelihood - sse(
                            foward_hidden,
                            hidden_list[layer_index]) / sigma_list[layer_index]
                        # print( sse(
                        #     foward_hidden,
                        #     hidden_list[layer_index]) / sigma_list[layer_index ])
                    else:
                        hidden_likelihood = hidden_likelihood - sse(
                            net.block_list[layer_index](hidden_list[layer_index - 1]),
                            hidden_list[layer_index]) / sigma_list[layer_index]
                        # print(sse(
                        #     net.block_list[layer_index](hidden_list[layer_index - 1]),
                        #     hidden_list[layer_index]) / sigma_list[layer_index ])

                    hidden_likelihood.backward()
                    step_proposal_lr = proposal_lr[layer_index]
                    gamma = alpha / step_proposal_lr
                    with torch.no_grad():
                        # momentum_list[layer_index] = (1 - alpha) * momentum_list[layer_index] + step_proposal_lr / 2 * \
                        #                              hidden_list[
                        #                                  layer_index].grad + torch.FloatTensor(
                        #     hidden_list[layer_index].shape).to(device).normal_().mul(
                        #     np.sqrt(alpha * step_proposal_lr * temperature[layer_index]))
                        # hidden_list[layer_index].data += momentum_list[layer_index]
                        momentum_list[layer_index] = (1 - gamma * step_proposal_lr) * momentum_list[layer_index] + step_proposal_lr * hidden_list[
                        layer_index].grad + torch.FloatTensor(
                        hidden_list[layer_index].shape).to(device).normal_().mul(
                        np.sqrt(2 * gamma * step_proposal_lr))
                        hidden_list[layer_index].data += step_proposal_lr * momentum_list[layer_index]

            # num_optim_step = 2000
            num_optim_step = 1
            for layer_index in range(num_block):
                for step in range(num_optim_step):
                    if layer_index == 0:
                        loss = sse(net.block_list[layer_index](x_train[subsample,]),
                                   hidden_list[layer_index]) / subn
                    elif layer_index == num_block - 1:
                        loss = loss_func_sum(net.block_list[layer_index](hidden_list[layer_index - 1]),
                                             y_train[subsample,]) / subn
                    else:
                        loss = sse(net.block_list[layer_index](hidden_list[layer_index - 1]),
                                   hidden_list[layer_index]) / subn
                    optimizer_list[layer_index].zero_grad()
                    loss.backward()
                    optimizer_list[layer_index].step()

        with torch.no_grad():
            if regression_flag:
                print('epoch: ', epoch)

                output, last_hidden_layer_train = net(x_train)
                train_loss = loss_func(output, y_train)
                train_loss_path[epoch] = train_loss
                train_corr = np.corrcoef(output.squeeze(), y_train.squeeze())[0,1]
                train_accuracy_path[epoch] = train_corr
                print("train loss: ", train_loss, "; train corr: ", train_corr)

                output, last_hidden_layer_test = net(x_test)
                test_loss = loss_func(output, y_test)
                test_loss_path[epoch] = test_loss
                test_corr = np.corrcoef(output.squeeze(), y_test.squeeze())[0,1]
                test_accuracy_path[epoch] = test_corr
                print("test loss: ", test_loss, "; test corr: ", test_corr)

            else:
                print('epoch: ', epoch)
                output, last_hidden_layer_train = net(x_train)
                train_loss = loss_func(output, y_train)
                prediction = output.data.max(1)[1]
                train_accuracy = prediction.eq(y_train.data).sum().item() / ntrain
                train_loss_path[epoch] = train_loss
                train_accuracy_path[epoch] = train_accuracy
                print("train loss: ", train_loss, 'train accuracy: ', train_accuracy)

                output, last_hidden_layer_test = net(x_test)
                test_loss = loss_func(output, y_test)
                prediction = output.data.max(1)[1]
                test_accuracy = prediction.eq(y_test.data).sum().item() / ntest
                test_loss_path[epoch] = test_loss
                test_accuracy_path[epoch] = test_accuracy
                print("test loss: ", test_loss, 'test accuracy: ', test_accuracy)

    # get the output of the last hidden layer
    output, last_hidden_layer_train = net(x_train)
    output, last_hidden_layer_test = net(x_test)


    # save the local networks
    import pickle

    f = open(PATH + "net_" + net_index + ".txt", 'wb')
    pickle.dump(net, f)
    f.close()

    
    f = open(PATH + 'loss_path_' + net_index + '.txt', 'wb')
    pickle.dump([train_loss_path, test_loss_path, train_accuracy_path, test_accuracy_path], f)
    f.close()

    return net, last_hidden_layer_train, last_hidden_layer_test