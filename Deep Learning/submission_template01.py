import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Свёрточные слои
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        # Полносвязные слои
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=256)  # 128 * 4 * 4 = 2048
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # conv1 -> BN -> ReLU -> pool1
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # conv2 -> BN -> ReLU -> pool2
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # conv3 -> BN -> ReLU -> pool3
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x)))           # fc1 -> ReLU -> Dropout
        x = self.fc2(x)
        return x

# Инициализация модели
model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Гиперпараметры
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Функция обучения (предполагается, что она уже определена)
def train(model, loss_fn, optimizer, n_epoch=3):
    # Ваш код для обучения модели
    return model

# Функция evaluate (предполагается, что она уже определена)
def evaluate(model, dataloader, loss_fn):
    # Ваш код для оценки модели
    return 0.0, 0.0

# Обучение
model = train(model, loss_fn, optimizer, n_epoch=10)  # Увеличиваем количество эпох до 10

# Тестирование
test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Accuracy на тесте:', test_accuracy)

# Сохранение модели
model.eval()
x = torch.randn((1, 3, 32, 32))
model_path = "model.pth"

# Проверка, что директория существует
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Сохранение модели
torch.jit.save(torch.jit.script(model.cpu()), model_path)
print(f"Модель успешно сохранена в {model_path}")

# Проверка, что файл существует и не пустой
assert os.path.exists(model_path), f"Файл {model_path} не существует!"
assert os.path.getsize(model_path) > 0, f"Файл {model_path} пуст!"

print("Все проверки пройдены. Файл модели корректен.")
