
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

class CNN(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size):
        super().__init__()
        pad_size = kernel_size - 1
        if pad_size % 2 == 0:
            self.padvals = (int(pad_size/2), int(pad_size/2))
        else:
            self.padvals = (ceil(pad_size / 2) - 1, ceil(pad_size / 2))
        10 - kernel_size
        self.conv1 = nn.Conv2d(in_dim, 16, kernel_size=(1, kernel_size), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(16, 4, kernel_size=(1, kernel_size))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.fc1 = nn.Linear(4*2, out_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(F.pad(x, self.padvals))))
        x = self.pool2(F.relu(self.conv2(F.pad(x, self.padvals))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

def make_model(in_dim, out_dim, **kwargs):
    model = CNN(in_dim, out_dim, **kwargs)
    return model