import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from model import *
from process_data import preprocess_data
from sklearn.preprocessing import MinMaxScaler
import argparse
import time
import os
import errno

# Basic Setting
parser = argparse.ArgumentParser(description='Running vanilla autoencoder')
parser.add_argument('--seed', default=1, type=int, help='set seed')
parser.add_argument('--data_path', default="./data/", type=str, help='folder name for loading data')
parser.add_argument('--base_path', default='./result/', type = str, help = 'base path for saving result')
parser.add_argument('--model_path', default='/', type=str, help='folder name for saving model')
parser.add_argument('--data_index', default=0, type=int)
parser.add_argument('--data_name', default='slice_localization', type=str, help='the name of the dataset')
parser.add_argument('--device', default='cpu', type=str, help='the device used to run the experiment')


# Training Setting
parser.add_argument('--num_epochs', default=20, type=int, help='total number of training epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum in SGD')
parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay in SGD')
parser.add_argument('--batch_size', default=800, type=int, help='batch size for training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate for the optimizer')
parser.add_argument('--net_architecture', default=[200, 192, 200], type=list, help='the architecture of the global stonet')
parser.add_argument('--cross_validate_index', default=0, type=int, help='specify which fold of 5 fold cross validation')

args = parser.parse_args()



seed = args.seed
device = args.device
num_epochs = args.num_epochs
weight_decay = args.weight_decay
momentum = args.momentum
batch_size = args.batch_size
lr = args.lr
net_architecture = args.net_architecture
data_name = args.data_name
cross_validate_index = args.cross_validate_index
net_architecture = args.net_architecture

for cross_validate_index in range(6, 10):
    # load data
    x_train, y_train, x_test, y_test = preprocess_data(data_name, cross_validate_index)
    dim = x_test.shape[1]
    reduce_dim = net_architecture[len(net_architecture)//2]
    net_architecture = [dim] + net_architecture
    net_architecture.append(dim)
    
    
    PATH = args.base_path + data_name + '/autoencoder/reduce_dim_' + str(reduce_dim) + '/cross_' + str(cross_validate_index) + '/'
 
    if not os.path.isdir(PATH):
            try:
                os.makedirs(PATH)
            except OSError as exc:  # Python >2.5
                if exc.errno == errno.EEXIST and os.path.isdir(PATH):
                    pass
                else:
                    raise
    
    # build local networks
    num_hidden = len(net_architecture)
    block_list = []
    for i in range(num_hidden - 1):
        block_list.append(nn.Sequential(nn.Linear(net_architecture[i], net_architecture[i+1]), nn.Tanh()))
    
    num_block = len(block_list)

    model = Net(block_list)
    
    # model = autoencoder(net_architecture)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
    if data_name == 'MNIST':
        x_train, x_test = x_train.view(x_train.size(0), -1), x_test.view(x_test.size(0), -1)
        x_train, x_test, y_train, y_test = x_train[:20000], x_test[:20000], y_train[:20000], y_test[:20000]
        
    ntrain = x_train.shape[0]
    index = np.arange(ntrain)
    train_loss_path = []
    test_loss_path = []
    
    start_time = time.perf_counter()
    for epoch in range(num_epochs):
        for iter_index in range(ntrain // batch_size):
            subsample = index[(iter_index * batch_size):((iter_index + 1) * batch_size)]
            img, img_test = x_train[subsample,], x_test
            img,img_test = img.view(img.size(0), -1), img_test.view(img_test.size(0), -1)
            img, img_test = Variable(img), Variable(img_test)
            # ===================forward=====================
            output, _ = model(img)
            output_test, _ = model(img_test)
            loss = criterion(output, img)
            loss_test = criterion(output_test, img_test)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], train loss:{:.4f}, test loss: {:.4f}'.format(epoch + 1, num_epochs, loss.data, loss_test.data))
        train_loss_path.append(loss)
        test_loss_path.append(loss_test)
    
    output, reduced_train = model(x_train)
    output, reduced_test = model(x_test)
    time_elapse = time.perf_counter() - start_time
    
    
    
    import pickle
    filename = PATH + 'time_elapse_reduce_dim_' + str(reduce_dim) + '.txt'
    f = open(filename, 'wb')
    pickle.dump(time_elapse, f)
    f.close()
    
    filename = PATH + 'loss_path_reduce_dim_' + str(reduce_dim) + '.txt'
    f = open(filename, 'wb')
    pickle.dump([train_loss_path, test_loss_path], f)
    f.close()
    
    filename = PATH + 'model_reduce_dim_' + str(reduce_dim) + '.txt'
    f = open(filename, 'wb')
    pickle.dump(model, f)
    f.close()
    
    filename = PATH + 'reduced_x_train_reduce_dim_' + str(reduce_dim) + '.txt'
    f = open(filename, 'wb')
    pickle.dump(reduced_train, f)
    f.close()
    
    filename = PATH + 'reduced_x_test_reduce_dim_' + str(reduce_dim) + '.txt'
    f = open(filename, 'wb')
    pickle.dump(reduced_test, f)
    f.close()
    
