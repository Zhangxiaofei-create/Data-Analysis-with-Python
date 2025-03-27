import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
 bin/sh ./doit.sh
tested_assignment.py
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Определяем слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Рассчитываем размер входного тензора для полносвязного слоя
        # После всех операций размер будет (5, 6, 6)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(5 * 6 * 6, 100)  # 180 входов
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # x: [batch_size, 3, 32, 32]
        
        # conv1: [32, 3, 32, 32] -> [32, 3, 28, 28]
        x = self.conv1(x)
        x = F.relu(x)  # используем F.relu
        
        # maxpool1: [32, 3, 28, 28] -> [32, 3, 14, 14]
        x = self.pool1(x)
        
        # conv2: [32, 3, 14, 14] -> [32, 5, 12, 12]
        x = self.conv2(x)
        x = F.relu(x)  # используем F.relu
        
        # maxpool2: [32, 5, 12, 12] -> [32, 5, 6, 6]
        x = self.pool2(x)
        
        # flatten: [32, 5, 6, 6] -> [32, 180]
        x = self.flatten(x)
        
        # fc1: [32, 180] -> [32, 100]
        x = self.fc1(x)
        x = F.relu(x)  # используем F.relu
        
        # fc2: [32, 100] -> [32, 10]
        x = self.fc2(x)
        
        return x
