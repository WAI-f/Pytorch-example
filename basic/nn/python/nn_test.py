# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


# basic model define
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # input channel:1; output channel:6; kernel size：3*3
        self.conv2 = nn.Conv2d(6, 16, 3)  # input channel:6; output channel:16; kernel size：3*3
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # full connect
        self.fc2 = nn.Linear(120, 84)  # full connect
        self.fc3 = nn.Linear(84, 10)  # full connect

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# view trainable parameters
params = list(net.parameters())
for i in range(len(params)):
    print(params[i].size())  # weight & bias

# random input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# calculate MSE Loss
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

# check conv1 bias gradients before and after the backward
net.zero_grad()
print(net.conv1.bias.grad)  # before backward
loss.backward()
print(net.conv1.bias.grad)  # after backward
