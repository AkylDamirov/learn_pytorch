#task #3
import torch
# import torch.nn as nn
#
#
# class MyModel(nn.Module):
#     def __init__(self, inp, out):
#         super().__init__()
#         self.conv = nn.Conv2d(inp, out, (3, 3), padding=1, bias=False)
#         self.batch = nn.BatchNorm2d(out)
#         self.relu = nn.ReLU()
#         self.globalpool = nn.AvgPool2d((10, 10))
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.batch(x)
#         x = self.relu(x)
#         x = self.globalpool(x)
#         return x
#
#
# model = MyModel(5, 10)
#
# input_tensor = torch.rand(1, 5, 10, 10)
# output = model(input_tensor)
#
# print(output.shape)
# print(output)

#4
# Импортируйте модуль
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.conv = nn.Conv2d(inp, 10, (3, 3), bias=False)
        self.batch = nn.BatchNorm2d(10)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d((8, 8))
        self.layer2 = nn.Linear(10 * 1 * 1, out)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.layer2(x)
        return out



model = MyModel(5, 5)
input_tensor = torch.rand(1, 5, 10, 10)
output = model(input_tensor)

print(output.shape)