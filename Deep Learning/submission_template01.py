import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоёв сети
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=180, out_features=100)  # 5 * 6 * 6 = 180
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        # Forward pass сети
        x = self.pool1(F.relu(self.conv1(x)))  # conv1 -> ReLU -> pool1
        x = self.pool2(F.relu(self.conv2(x)))  # conv2 -> ReLU -> pool2
        x = self.flatten(x)                    # flatten
        x = F.relu(self.fc1(x))                # fc1 -> ReLU
        x = self.fc2(x)                        # fc2
        return x

# Проверка реализации
if __name__ == "__main__":
    img = torch.Tensor(np.random.random((32, 3, 32, 32)))
    model = ConvNet()
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

    # pool2
    assert model.pool2.kernel_size == 2, "неверный размер ядра у pool2"

    # fc1
    assert model.fc1.out_features == 100, "неверный размер out_features у fc1"
    # fc2
    assert model.fc2.out_features == 10, "неверный размер out_features у fc2"

    print("Все проверки пройдены успешно!")

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

# Обучение
model = train(model, loss_fn, optimizer, n_epoch=10)  # Увеличиваем количество эпох до 10

# Тестирование
test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Accuracy на тесте:', test_accuracy)

model.eval()
x = torch.randn((1, 3, 32, 32))
torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model.pth")
