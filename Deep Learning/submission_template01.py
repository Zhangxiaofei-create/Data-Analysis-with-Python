import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Создаем модель
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Linear(150, 256)  # 10x15 = 150 пикселей
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)     # 6 классов

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Создаем экземпляр модели
model = SimpleNet()

# Сохраняем модель в файл
torch.save(model.state_dict(), 'model.pth')

# Для проверки корректности сохранения можно загрузить модель:
# model = SimpleNet()
# model.load_state_dict(torch.load('model.pth'))
# model.eval()

# Дополнительные функции для работы с данными
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

# Пример использования:
# trainset = torchvision.datasets.MNIST(root='.', train=True, download=True, transform=transform)
# trainloader = create_dataloader(trainset)
