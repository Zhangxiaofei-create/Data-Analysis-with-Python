import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import numpy as np
import os

#  Загружаем данные CIFAR-10
train_data = datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=transforms.ToTensor())

#  Разделение на train и validation
train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

#  Улучшенная модель
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.fc2(x)
        return x

#  Функции обучения и оценки
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    num_correct = 0
    num_elements = 0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)
            num_correct += torch.sum(y_pred == y_batch)
            num_elements += len(y_batch)

    accuracy = num_correct / num_elements
    return accuracy.item(), np.mean(losses)

def train(model, loss_fn, optimizer, train_loader, val_loader, device, n_epoch=10):
    for epoch in range(n_epoch):
        model.train()
        running_losses = []
        running_accuracies = []

        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            running_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = (model_answers == y_batch).float().mean().item()
            running_accuracies.append(train_accuracy)

            if (i+1) % 100 == 0:
                print(f"Итерация {i+1}: loss={np.mean(running_losses)}, accuracy={np.mean(running_accuracies)}")

        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Эпоха {epoch+1}: val loss={val_loss}, val accuracy={val_accuracy}")

    return model

#  Обучение
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Запуск обучения
model = train(model, loss_fn, optimizer, train_loader, val_loader, device, n_epoch=10)

#  Тестирование
test_accuracy, _ = evaluate(model, test_loader, loss_fn, device)
print(f'Accuracy на тесте: {test_accuracy:.4f}')

#  Сохранение модели (исправленный вариант)
model.eval()  # Переводим в режим оценки перед сохранением
model_path = "model.pth"

# Проверка, что директория существует
os.makedirs(os.path.dirname(model_path), exist_ok=True)

#  Используем torch.save() вместо torch.jit.script()
torch.save(model.state_dict(), model_path)
print(f"Модель успешно сохранена в {model_path}")

#  Проверка, что файл существует и не пустой
assert os.path.exists(model_path), f"Файл {model_path} не существует!"
assert os.path.getsize(model_path) > 0, f"Файл {model_path} пуст!"

print("Все проверки пройдены. Файл модели корректен.")

# Загрузка модели
loaded_model = ConvNet().to(device)
loaded_model.load_state_dict(torch.load(model_path, map_location=device))
loaded_model.eval()
print("Модель успешно загружена.")

#  Проверка работы загруженной модели
x = torch.randn((1, 3, 32, 32)).to(device)
output = loaded_model(x)
print("Выход модели после загрузки:", output)
