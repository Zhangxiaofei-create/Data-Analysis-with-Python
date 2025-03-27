import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from IPython.display import clear_output

# 数据加载
train_data = datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=transforms.ToTensor())

train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

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
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 评估函数
def evaluate(model, dataloader, loss_fn):
    model.eval()  # 设置模型为评估模式
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
            num_correct += torch.sum(y_pred == y_batch).item()
            num_elements += y_batch.size(0)

    accuracy = num_correct / num_elements
    return accuracy, np.mean(losses)

# 训练函数
def train(model, loss_fn, optimizer, n_epoch=3):
    for epoch in range(n_epoch):
        model.train()  # 设置模型为训练模式
        running_losses = []
        running_accuracies = []

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()  # 梯度清零
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_losses.append(loss.item())
            y_pred = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_pred == y_batch).item() / y_batch.size(0)
            running_accuracies.append(train_accuracy)

        # 每个epoch结束后评估验证集
        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1}/{n_epoch}, Train Loss: {np.mean(running_losses):.4f}, "
              f"Train Accuracy: {np.mean(running_accuracies):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model

model = train(model, loss_fn, optimizer, n_epoch=20)

# 测试集评估
test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print(f'Test Accuracy: {test_accuracy:.4f}')

# 保存模型
print("开始保存模型...") # 添加打印语句
model.eval()
model_cpu = model.cpu() # 确保模型在 CPU 上
x = torch.randn((1, 3, 32, 32)) # 输入也使用 CPU
# 保存模型
model.eval()
x = torch.randn((1, 3, 32, 32)).to(device)
try:
    torch.jit.save(torch.jit.trace(model.cpu(), (x.cpu())), "model.pth")
    print("模型保存成功！")
except Exception as e:
    print("模型保存失败：", str(e))
try:
    loaded_model = torch.jit.load("model.pth")
    print("保存后立即加载模型成功!")
    # 可以添加一些简单的模型结构或输出的检查，确保加载的模型是预期的
except Exception as e:
    print(f"保存后立即加载模型失败，错误信息: {e}")



