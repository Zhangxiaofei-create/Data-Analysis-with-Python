class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(3, 5, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(5 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Forward pass сети
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Определение слоев сети
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(12 * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # Forward pass сети
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

# Объявление модели
model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Выбор функции потерь и алгоритма оптимизации
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
model = train(model, loss_fn, optimizer, n_epoch=10)

test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Accuracy на тесте', test_accuracy)
model.eval()
x = torch.randn((1, 3, 32, 32))
torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model.pth")
