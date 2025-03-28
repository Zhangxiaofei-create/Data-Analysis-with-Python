import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        # DEFINE LAYERS OF THE NETWORK

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=(3, 3))
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten() # equivalent to x.view(x.shape[0], -1)

        self.fc1 = nn.Linear(5 * 5 * 5, 100)  # Corrected the input size to 5*5*5
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # x dimension ~ [64, 3, 32, 32]

        # IMPLEMENT FORWARD PASS OF THE NETWORK

        x = self.conv1(x)  # conv1
        x = F.relu(x)       # ReLU activation
        x = self.pool1(x)  # maxpool1

        x = self.conv2(x)  # conv2
        x = F.relu(x)       # ReLU activation
        x = self.pool2(x)  # maxpool2

        x = self.flatten(x)

        x = self.fc1(x)    # fc1
        x = F.relu(x)       # ReLU activation
        x = self.fc2(x)    # fc2

        return x

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

import torchvision
from torchvision import datasets, transforms

from matplotlib import pyplot as plt
from IPython.display import clear_output

# Load data (same as provided)
train_data = datasets.CIFAR10(root="./cifar10_data", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.CIFAR10(root="./cifar10_data", train=False, download=True, transform=transforms.ToTensor())

train_size = int(len(train_data) * 0.8)
val_size = len(train_data) - train_size

train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

def show_images(images, labels):
    f, axes= plt.subplots(1, 10, figsize=(30,5))

    for i, axis in enumerate(axes):
        img = images[i].numpy()
        img = np.transpose(img, (1, 2, 0))

        axes[i].imshow(img)
        axes[i].set_title(labels[i].numpy())

    plt.show()

for batch in train_loader:
    images, labels = batch
    break

show_images(images, labels)

# YOUR CODE HERE
# DECLARE CONVOLUTIONAL NEURAL NETWORK CLASS
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # More filters, padding
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # More filters, padding
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Added conv layer
        self.bn3 = nn.BatchNorm2d(64)  # BatchNorm
        self.pool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # Increased hidden size
        self.bn4 = nn.BatchNorm1d(512)  # BatchNorm
        self.dropout = nn.Dropout(0.5)  # Dropout
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # Added layer
        x = self.flatten(x)
        x = F.relu(self.bn4(self.fc1(x)))  # Increased hidden size
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ConvNet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def evaluate(model, dataloader, loss_fn):

    losses = []

    num_correct = 0
    num_elements = 0

    for i, batch in enumerate(dataloader):

        X_batch, y_batch = batch
        num_elements += len(y_batch)

        with torch.no_grad():
            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            losses.append(loss.item())

            y_pred = torch.argmax(logits, dim=1)

            num_correct += torch.sum(y_pred.cpu() == y_batch.cpu()) #Fixed this line

    accuracy = num_correct / num_elements

    return accuracy.numpy(), np.mean(losses)

def train(model, loss_fn, optimizer, n_epoch=3):

    for epoch in range(n_epoch):

        print("Epoch:", epoch+1)

        model.train(True)

        running_losses = []
        running_accuracies = []
        for i, batch in enumerate(train_loader):
            X_batch, y_batch = batch

            logits = model(X_batch.to(device))

            loss = loss_fn(logits, y_batch.to(device))
            running_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model_answers = torch.argmax(logits, dim=1)
            train_accuracy = torch.sum(y_batch == model_answers.cpu()) / len(y_batch)
            running_accuracies.append(train_accuracy)

            if (i+1) % 100 == 0:
                print("Average train loss and accuracy over the last 50 iterations:",
                      np.mean(running_losses), np.mean(running_accuracies), end='\n')

        model.train(False)

        val_accuracy, val_loss = evaluate(model, val_loader, loss_fn=loss_fn)
        print("Epoch {}/{}: val loss and accuracy:".format(epoch+1, n_epoch,),
                      val_loss, val_accuracy, end='\n')

    return model

model = ConvNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model = train(model, loss_fn, optimizer, n_epoch=10) #Increase number of epochs

test_accuracy, _ = evaluate(model, test_loader, loss_fn)
print('Accuracy on test', test_accuracy)

if test_accuracy <= 0.5:
    print("Quality on the test is below 0.5, 0 points")
elif test_accuracy < 0.6:
    print("Quality on the test is between 0.5 and 0.6, 0.5 points")
elif test_accuracy >= 0.6:
    print("Quality on the test is above 0.6, 1 point")

model.eval()
x = torch.randn((1, 3, 32, 32))
torch.jit.save(torch.jit.trace(model.cpu(), (x)), "model.pth")
