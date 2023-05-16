import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.utils.data
from sklearn.preprocessing import MinMaxScaler
from model import Net, Resize

def preprocess_data(data_name, cross_validate_index):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    if data_name == 'measurement_error':
        TotalP = 5
        NTrain = 500
        x_train = np.matrix(np.zeros([NTrain, TotalP]))
        y_train = np.matrix(np.zeros([NTrain, 1]))

        sigma = 1.0
        for i in range(NTrain):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)
                # x_train[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / 2.0
            x0 = x_train[i, 0]
            x1 = x_train[i, 1]
            x2 = x_train[i, 2]
            x3 = x_train[i, 3]
            x4 = x_train[i, 4]

            y_train[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

            x_train[i, 0] = x_train[i, 0] + np.random.normal(0, 0.5)
            x_train[i, 1] = x_train[i, 1] + np.random.normal(0, 0.5)
            x_train[i, 2] = x_train[i, 2] + np.random.normal(0, 0.5)
            x_train[i, 3] = x_train[i, 3] + np.random.normal(0, 0.5)
            x_train[i, 4] = x_train[i, 4] + np.random.normal(0, 0.5)

        NTest = 500
        x_test = np.matrix(np.zeros([NTest, TotalP]))
        y_test = np.matrix(np.zeros([NTest, 1]))

        for i in range(NTest):
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(2.0)
                # x_test[i, j] = (ee + np.sqrt(sigma) * np.random.normal(0, 1)) / 2.0
            x0 = x_test[i, 0]
            x1 = x_test[i, 1]
            x2 = x_test[i, 2]
            x3 = x_test[i, 3]
            x4 = x_test[i, 4]

            y_test[i, 0] = 5 * x1 / (1 + x0 * x0) + 5 * np.sin(x2 * x3) + 2 * x4 + np.random.normal(0, 1)

            x_test[i, 0] = x_test[i, 0] + np.random.normal(0, 0.5)
            x_test[i, 1] = x_test[i, 1] + np.random.normal(0, 0.5)
            x_test[i, 2] = x_test[i, 2] + np.random.normal(0, 0.5)
            x_test[i, 3] = x_test[i, 3] + np.random.normal(0, 0.5)
            x_test[i, 4] = x_test[i, 4] + np.random.normal(0, 0.5)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)

        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    if data_name == 'full_row_rank':
        TotalP = 1000
        a = 1
        b = 1
        W1 = np.matrix(np.random.choice([-2, -1, 1, 2], size=TotalP * 5, replace=True).reshape([TotalP, 5]))
        W2 = np.matrix(np.random.choice([-2, -1, 1, 2], size=5 * 5, replace=True).reshape([5, 5]))
        W3 = np.matrix(np.random.choice([-2, -1, 1, 2], size=5 * 1, replace=True).reshape([5, 1]))
        NTrain = 1000
        x_train = np.matrix(np.zeros([NTrain, TotalP]))
        y_train = np.matrix(np.zeros([NTrain, 1]))
        sigma = 1.0
        for i in range(NTrain):
            if i % 1000 == 0:
                print("x_train generate = ", i)
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_train[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)

        temp = np.tanh(x_train * W1)
        temp = np.tanh(temp * W2)
        y_train = temp * W3 + np.random.normal(0, 1, size=y_train.shape)

        NTest = 1000
        x_test = np.matrix(np.zeros([NTest, TotalP]))
        y_test = np.matrix(np.zeros([NTest, 1]))

        sigma = 1.0
        for i in range(NTest):
            if i % 1000 == 0:
                print("x_test generate = ", i)
            ee = np.sqrt(sigma) * np.random.normal(0, 1)
            for j in range(TotalP):
                x_test[i, j] = (a * ee + b * np.sqrt(sigma) * np.random.normal(0, 1)) / np.sqrt(a * a + b * b)

        temp = np.tanh(x_test * W1)
        temp = np.tanh(temp * W2)
        y_test = temp * W3 + np.random.normal(0, 1, size=y_test.shape)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)

        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)

    elif data_name == 'ads':
        temp = pd.read_table('/home/liang257/PycharmProjects/StoNet_SDR/data/ads/ad.data', sep = ',', header=None)
        temp = np.mat(temp)
        temp = temp[:,4:]

        x_data = temp[:, 0:(-1)].astype('float32')
        y_data = (temp[:,-1] == 'ad.')
        y_data = np.array(y_data.astype('int')).reshape(y_data.shape[0])


        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)


        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'parkinson':
        temp = pd.read_table('/home/liang257/PycharmProjects/StoNet_SDR/data/prakinson_telemonitoring/parkinsons_updrs.data', sep=',')

        temp = np.mat(temp)
        x_data = temp[:, 6:]
        y_data = temp[:, 5]

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])
        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        y_train_mean = np.mean(y_train)
        y_train_std = np.std(y_train)

        y_train = (y_train - y_train_mean) / y_train_std

        y_test = (y_test - y_train_mean) / y_train_std

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
    elif data_name == 'qsar':

        temp = pd.read_csv('/home/liang257/PycharmProjects/StoNet_SDR/data/qsar/qsar_androgen_receptor.csv', sep=';', header=None)
        temp = np.mat(temp)
        x_data = temp[:, 0:-1].astype('float64')
        y_data = temp[:, -1]

        y_data = (y_data == 'positive')

        y_data = np.array(y_data.astype('int')).reshape(y_data.shape[0])

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)


        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)


    elif data_name == 'EOMI':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/EA/ea.txt'
        temp = pd.read_table(data_file_name, sep='\t')
        temp = np.mat(temp)
        x_data = temp[:, 4:(-1)].transpose().astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/EA/class.csv'
        temp = pd.read_csv(data_file_name, sep='\t')
        temp = np.mat(temp)
        y_data = temp[:, 1]
        y_data = np.array(y_data.astype('int')).reshape(y_data.shape[0])

        permutation = np.random.choice(range(x_data.shape[0]), x_data.shape[0], replace=False)
        size_test = np.round(x_data.shape[0] * 0.2).astype(int)
        divid_index = np.arange(x_data.shape[0])

        lower_bound = cross_validate_index * size_test
        upper_bound = (cross_validate_index + 1) * size_test
        test_index = (divid_index >= lower_bound) * (divid_index < upper_bound)

        index_train = permutation[[not _ for _ in test_index]]
        index_test = permutation[test_index]

        x_train = x_data[index_train, :]
        y_train = y_data[index_train]

        x_test = x_data[index_test, :]
        y_test = y_data[index_test]

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)


    elif data_name == 'MNIST':

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        x_train = train_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)
        x_test = test_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)


        # x_train = x_train.reshape([x_train.shape[0], -1])
        # x_test = x_test.reshape([x_test.shape[0], -1])


        x_train = x_train.reshape([x_train.shape[0], 1, 28, 28])
        x_test = x_test.reshape([x_test.shape[0], 1, 28, 28])

        y_train = train_set.targets
        y_test = test_set.targets

        print('here')
        x_train = x_train.to(device)
        print('here1')
        y_train = y_train.to(device)
        print('here2')
        x_test = x_test.to(device)
        print('here3')
        y_test = y_test.to(device)
        print('here4')

    elif data_name == 'MNIST_transfer':

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        x_train = train_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)
        x_test = test_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)


        # x_train = x_train.reshape([x_train.shape[0], -1])
        # x_test = x_test.reshape([x_test.shape[0], -1])


        x_train = x_train.reshape([x_train.shape[0], 1, 28, 28])
        x_test = x_test.reshape([x_test.shape[0], 1, 28, 28])

        y_train = train_set.targets
        y_test = test_set.targets

        print('here')
        x_train = x_train.to(device)
        print('here1')
        y_train = y_train.to(device)
        print('here2')
        x_test = x_test.to(device)
        print('here3')
        y_test = y_test.to(device)
        print('here4')

        block_list = []
        block_list.append(nn.Conv2d(1, 20, 5, 1))
        block_list.append(nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Conv2d(20, 50, 5, 1)))
        block_list.append(nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), Resize(), nn.Linear(4 * 4 * 50, 500)))
        block_list.append(nn.Sequential(nn.ReLU(), nn.Linear(500, 10)))


        net = Net(block_list)
        net.to(device)

        net.load_state_dict(torch.load('/scratch/gilbreth/sun748/stonet_new/sgd/MNIST/test_run/model299.pt'))
        # with torch.no_grad():
        #     for i in range(2):
        #         x_train = net.block_list[i](x_train)
        #         x_test = net.block_list[i](x_test)
        #     temp = nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), Resize())
        #     x_train = temp(x_train)
        #     x_test = temp(x_test)
        with torch.no_grad():
            for i in range(3):
                x_train = net.block_list[i](x_train)
                x_test = net.block_list[i](x_test)
            # temp = nn.Sequential(nn.ReLU(), nn.MaxPool2d((2, 2)), Resize())
            # x_train = temp(x_train)
            # x_test = temp(x_test)



    elif data_name == 'CIFAR10':

        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        train_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             normalize])

        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        x_train = torch.FloatTensor(train_set.data).div(255.0)
        x_test = torch.FloatTensor(test_set.data).div(255.0)

        mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
        std = torch.FloatTensor([0.2470, 0.2435, 0.2616])
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        # x_train = train_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)
        # x_test = test_set.data.type(torch.FloatTensor).div(255).sub(0.1307).div(0.3081).unsqueeze(1)


        x_train = x_train.reshape([x_train.shape[0], -1])
        x_test = x_test.reshape([x_test.shape[0], -1])

        y_train = torch.LongTensor(train_set.targets)
        y_test = torch.LongTensor(test_set.targets)

        print('here')
        x_train = x_train.to(device)
        print(x_train.shape)
        print('here1')
        y_train = y_train.to(device)
        print('here2')
        x_test = x_test.to(device)
        print('here3')
        y_test = y_test.to(device)
        print('here4')

    elif data_name == 'breast cancer':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/breast-cancer/breast-cancer_train_data_'+str(cross_validate_index)+'.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/breast-cancer/breast-cancer_train_labels_'+str(cross_validate_index)+'.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/breast-cancer/breast-cancer_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/breast-cancer/breast-cancer_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'flare solar':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/flare-solar/flare-solar_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/flare-solar/flare-solar_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/flare-solar/flare-solar_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/flare-solar/flare-solar_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'german':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/german/german_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/german/german_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/german/german_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/german/german_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'heart':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/heart/heart_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/heart/heart_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/heart/heart_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/heart/heart_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)


    elif data_name == 'ringnorm':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/ringnorm/ringnorm_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/ringnorm/ringnorm_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/ringnorm/ringnorm_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/ringnorm/ringnorm_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)


    elif data_name == 'thyroid':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/thyroid/thyroid_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/thyroid/thyroid_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/thyroid/thyroid_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep='  ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/thyroid/thyroid_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'twonorm':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/twonorm/twonorm_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep=' ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/twonorm/twonorm_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep=' ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/twonorm/twonorm_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep=' ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/twonorm/twonorm_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep=' ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == 'waveform':
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/waveform/waveform_train_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep=' ', header=None, engine='python')
        x_train = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/waveform/waveform_train_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep=' ', header=None, engine='python')
        y_train = np.mat(temp)
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/waveform/waveform_test_data_' + str(cross_validate_index) + '.asc'
        temp = pd.read_table(data_file_name, sep=' ', header=None, engine='python')
        x_test = np.mat(temp).astype('float64')

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/waveform/waveform_test_labels_' + str(cross_validate_index) + '.asc'
        temp = pd.read_csv(data_file_name, sep=' ', header=None, engine='python')
        y_test = np.mat(temp)
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)

    elif data_name == "breast_cancer":
        data = load_breast_cancer()
        # data.keys()
        data_X = pd.DataFrame(data.data, columns = data.feature_names)
        data_y = data.target
        scaler = MinMaxScaler()
        scaler.fit(data_X)
        scaler.transform(data_X)
        
        x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = cross_validate_index) 
        
        x_train = torch.FloatTensor(x_train.values).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test.values).to(device)
        y_test = torch.LongTensor(y_test).to(device)
        
    elif data_name == "parkinsons":
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/parkinsons.data'
        temp = pd.read_csv(data_file_name, engine='python')
        # y = temp['motor_UPDRS']
        data_y = temp['total_UPDRS']
        data_X = temp.drop(columns = ['motor_UPDRS','total_UPDRS'])
        scaler = MinMaxScaler()
        scaler.fit(data_X)
        data_X = scaler.transform(data_X)
        data_y = data_y.values.reshape(-1,1)
        scaler.fit(data_y)
        data_y = scaler.transform(data_y)
        
        x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = cross_validate_index) 
        
        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train.reshape(-1, 1)).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test.reshape(-1, 1)).to(device)
        
    elif data_name == "covtype":
        data_file_name = "/home/liang257/PycharmProjects/StoNet_SDR/data/covtype.data"
        temp = pd.read_csv(data_file_name, header=None, engine='python')
        # y = temp['motor_UPDRS']
        data_y = temp.iloc[:,-1]
        data_X = temp.iloc[:,:54]
        scaler = MinMaxScaler()
        scaler.fit(data_X.iloc[:,:9])
        data_X.iloc[:,:9] = scaler.transform(data_X.iloc[:,:9])
        
        x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = cross_validate_index) 
        y_train, y_test = y_train - 1, y_test - 1
        
        x_train = torch.FloatTensor(x_train.values).to(device)
        y_train = torch.LongTensor(y_train.values.squeeze()).to(device)
        x_test = torch.FloatTensor(x_test.values).to(device)
        y_test = torch.LongTensor(y_test.values.squeeze()).to(device)
        
    elif data_name == "UJIndoorLoc":
        PATH = "/home/liang257/PycharmProjects/StoNet_SDR/data/UJIndoorLoc/"
        train = pd.read_csv(PATH+"trainingData.csv")
        test = pd.read_csv(PATH+"validationData.csv")
        train_y = train["FLOOR"]
        test_y = test["FLOOR"]
        train_x = train.drop(columns=["FLOOR","BUILDINGID","LONGITUDE","LATITUDE"])
        test_x = test.drop(columns=["FLOOR","BUILDINGID","LONGITUDE","LATITUDE"])
    
        scaler = MinMaxScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)
        
        x_train = torch.FloatTensor(train_x).to(device)
        y_train = torch.LongTensor(train_y.values.squeeze()).to(device)
        x_test = torch.FloatTensor(test_x).to(device)
        y_test = torch.LongTensor(test_y.values.squeeze()).to(device)
        
    elif data_name == "slice_localization":
        filename = "/home/liang257/PycharmProjects/StoNet_SDR/data/slice_localization_data.csv"
        data = pd.read_csv(filename)
        data_y = data["reference"]
        data_X = data.drop(columns=["reference"])
        
        
        x_train, x_test, y_train, y_test = train_test_split(data_X, data_y, test_size = 0.2, random_state = cross_validate_index) 
        
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        
        y_train = y_train.values.reshape((-1,1))
        y_test = y_test.values.reshape((-1,1))
        scaler = MinMaxScaler()
        scaler.fit(y_train)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
        
        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
    elif data_name == "Arcene":
        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/Arcene/arcene_train.data'
        x_train = pd.read_table(data_file_name, sep=' ', header=None)
        x_train = x_train.values[:,:10000]

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/Arcene/arcene_train.labels'
        y_train = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_train = np.array(y_train.astype('int')).reshape(y_train.shape[0])
        y_train = (y_train + 1) / 2

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/Arcene/arcene_valid.data'
        x_test = pd.read_table(data_file_name, sep=' ', header=None, engine='python')
        x_test = x_test.values[:,:10000]

        data_file_name = '/home/liang257/PycharmProjects/StoNet_SDR/data/Arcene/arcene_valid.labels'
        y_test = pd.read_csv(data_file_name, sep='  ', header=None, engine='python')
        y_test = np.array(y_test.astype('int')).reshape(y_test.shape[0])
        y_test = (y_test + 1) / 2

        x_train_std = np.std(x_train, 0)
        x_train_std[x_train_std == 0] = 1
        x_train_mean = np.mean(x_train, 0)

        x_train = (x_train - np.full(x_train.shape, x_train_mean)) / np.full(x_train.shape, x_train_std)

        x_test = (x_test - np.full(x_test.shape, x_train_mean)) / np.full(x_test.shape, x_train_std)

        x_train = torch.FloatTensor(x_train).to(device)
        y_train = torch.LongTensor(y_train).to(device)
        x_test = torch.FloatTensor(x_test).to(device)
        y_test = torch.LongTensor(y_test).to(device)
        
    return x_train, y_train, x_test, y_test