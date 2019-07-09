import numpy as np
from numpy import asarray as ar
from numpy import sqrt
from numpy.random import rand, randn
import torch
import torch.nn as nn
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, length):
        super(CNN, self).__init__()
        # Cin = 1, Cout = 256, Kernel_size = 11
        self.conv1 = nn.Conv1d(1, 64, 3, stride=1, padding=1)
        # Cin = 256, Cout = 256, Kernel_size = 5
        self.conv2 = nn.Conv1d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 128, 3, stride=1, padding=1)

        # Batch Nromalization
        self.batnorm1 = nn.BatchNorm1d(64)
        self.batnorm2 = nn.BatchNorm1d(128)
        self.batnorm3 = nn.BatchNorm1d(128)

        self.relu = nn.LeakyReLU(0.01, True)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(p=0.5)
        self.len = length

        self.fc1 = nn.Linear(int(self.len / 8) * 128, 128)

        #self.fc1 = nn.Linear(self.len, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.conv1(x)          # Cin = 1, Cout = 64, Kernel_size = 11
        x = self.batnorm1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        x = self.conv3(x)          # Cin = 64, Cout = 128, Kernel_size = 5
        x = self.batnorm3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        x = x.view(-1, int(self.len / 8) * 128)
        #x = x.squeeze(1)
        x = self.fc1(x)            # Din = 16*256, Dout = 1024
        x = self.relu(x)
        x = self.fc2(x)            # Din = 1024, Dout = 1024
        x = self.relu(x)
        x = self.fc3(x)            # Din = 1024, Dout = 1

        return x
