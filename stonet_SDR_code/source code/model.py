import torch
import torch.nn as nn
import torch.nn.functional as F

class Resize(nn.Module):
    def __init__(self):
        super(Resize, self).__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x


class Net(nn.Module):
    def __init__(self, block_list):
        super(Net, self).__init__()
        self.block_list = block_list
        for i in range(len(block_list)):
            self.add_module('block' + str(i), block_list[i])
    def forward(self, x):
        for i in range(len(self.block_list)):
            last_hidden_layer = x
            x = self.block_list[i](x)
        return x, last_hidden_layer
    

# Define class Net
class DNN(nn.Module):
    def __init__(self, block_list):
        super(DNN, self).__init__()
        self.block_list = block_list
        for i in range(len(block_list)):
            self.add_module('block' + str(i), block_list[i])
    def forward(self, x):
        for i in range(len(self.block_list)):
            x = self.block_list[i](x)
        return x
 
    
# class stonet(nn.Module):
#     def __init__(self, dim):
#         super(stonet, self).__init__()
#         self.fc1 = nn.Linear(dim, 100)
#         self.fc2 = nn.Linear(100, 27)
#         self.fc3 = nn.Linear(27, 10)

#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         x = self.fc3(x)
#         return x
    

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        # encoder
        self.enc1 = nn.Linear(in_features=385, out_features=200)
        self.enc2 = nn.Linear(in_features=200, out_features=192)
        # decoder
        self.dec1 = nn.Linear(in_features=192, out_features=200)
        self.dec2 = nn.Linear(in_features=200, out_features=385)

    def forward(self, x):
        x = torch.tanh(self.enc1(x))
        reduced_data = torch.tanh(self.enc2(x))

        x = torch.tanh(self.dec1(reduced_data))
        x = torch.tanh(self.dec2(x))

        return x, reduced_data
    
