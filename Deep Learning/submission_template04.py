import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Определите слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)  # Здесь 5 * 6 * 6 - из-за размерности входов
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # Реализуйте forward pass сети
        x = self.pool1(F.relu(self.conv1(x)))  # Применяем conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x)))  # Применяем conv2 -> ReLU -> pool2
        x = self.flatten(x)                      # Преобразуем в плоский вектор
        x = F.relu(self.fc1(x))                 # Применяем fc1 -> ReLU
        x = self.fc2(x)                         # Применяем fc2
        return x

# Проверка реализации
img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)

# Проверка архитектуры
# conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool1"

# conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

# pool2
assert model.pool2.kernel_size == (2, 2), "неверный размер ядра у pool2"

# fc1
assert model.fc1.out_features == 100, "неверный размер out_features у fc1"

# fc2
assert model.fc2.out_features == 10, "неверный размер out_features у fc2"

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Сверточные слои
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

        # Dropout для регуляризации
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Проход вперед
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Max pool с уменьшением размерности
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Приведение тензора к нужному размеру для полносвязного слоя
        x = x.view(x.size(0), -1)  # Вытягиваем тензор в одну строку

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# Теперь создаем модель и переносим её на устройство
model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Далее идёт этап обучения и тестирования, который уже приведен в вашем коде
