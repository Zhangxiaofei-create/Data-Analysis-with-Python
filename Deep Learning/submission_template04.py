import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # [B,32,16,16]
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # [B,64,8,8]

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

img = torch.Tensor(np.random.random((32, 3, 32, 32)))
model = ConvNet()
out = model(img)
print("Output shape:", out.shape)  # [32, 10]

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # сверточные слои с большим числом фильтров + BatchNorm
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # полносвязные слои (добавим Dropout для регуляризации)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) # 32x16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x)))) # 64x8x8

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

# 训练完成后，保存模型
model.eval()
scripted_model = torch.jit.script(model.cpu())
scripted_model.save("model.pth")

