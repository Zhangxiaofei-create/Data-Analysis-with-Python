import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        # Исправлено: правильное количество входных признаков для fc1
        self.fc1 = nn.Linear(in_features=180, out_features=100)  # 5 * 6 * 6 = 180
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # Forward pass сети
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model():
    return MyModel()

# Проверка реализации
if __name__ == "__main__":
    img = torch.Tensor(np.random.random((32, 3, 32, 32)))
    model = create_model()
    out = model(img)

    # Проверка архитектуры
    # conv1
    assert model.conv1.kernel_size == (5, 5), "неверный размер ядра у conv1"
    assert model.conv1.in_channels == 3, "неверный размер in_channels у conv1"
    assert model.conv1.out_channels == 3, "неверный размер out_channels у conv1"

    # pool1
    assert model.pool1.kernel_size == 2, "неверный размер ядра у pool1"

    # conv2
    assert model.conv2.kernel_size == (3, 3), "неверный размер ядра у conv2"
    assert model.conv2.in_channels == 3, "неверный размер in_channels у conv2"
    assert model.conv2.out_channels == 5, "неверный размер out_channels у conv2"

    # fc1
    assert model.fc1.in_features == 180, "неверное количество входных признаков у fc1"

    print("Все проверки пройдены успешно!")
