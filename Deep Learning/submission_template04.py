import numpy as np
import torch
from torch import nn
import torch.nn.functional as F  # Добавлен импорт для F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Определите слои сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # Исправлено на MaxPool2d
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=5 * 6 * 6, out_features=100)  # Входные размеры нужно проверить и скорректировать
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
model = MyModel()  # Исправлено на MyModel
out = model(img)

# Проверка архитектуры
# conv1
assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

# pool1
assert model.pool1.kernel_size == (2, 2), "неверный размер ядра у pool1"  # Исправлено на (2, 2)

# conv2
assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"  # Исправлено

print("Все проверки пройдены успешно!")
