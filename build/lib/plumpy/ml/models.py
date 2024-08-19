
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
        self.conv1 = nn.Conv2d(in_dim, 8, kernel_size=(1, kernel_size), stride=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.conv2 = nn.Conv2d(8, 2, kernel_size=(1, kernel_size))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.fc1 = nn.Linear(2*10, out_dim)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(F.pad(x, self.padvals))))
        x = self.pool2(F.relu(self.conv2(F.pad(x, self.padvals))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x

class LogReg(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SmallResNet(nn.Module):
    def __init__(self, block, num_blocks, num_in, num_classes=10):
        super(SmallResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(num_in, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.squeeze(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def make_model(in_dim, out_dim, **kwargs):
    #model = CNN(in_dim, out_dim, **kwargs)
    #model = SmallResNet(BasicBlock, [1, 1], num_in=in_dim, num_classes=out_dim)
    model = LogReg(in_dim)
    return model