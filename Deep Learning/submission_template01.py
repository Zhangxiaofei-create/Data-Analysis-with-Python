# model.py（模型定义与保存）
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()  # 修复缩进
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [B,32,16,16]
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [B,64,8,8]
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 保存模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    model.eval()
    
    # 使用 torch.jit.script 需要模型在 CPU 上
    scripted_model = torch.jit.script(model.cpu())
    scripted_model.save("scripted_model.pth")  # 明确文件名
