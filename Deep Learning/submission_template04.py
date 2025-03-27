import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

# Create and save the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)

# Save using state_dict (more reliable)
torch.save(model.state_dict(), "model_state_dict.pth")

# Or save using torch.jit.script (make sure to put model in eval mode)
model.eval()
scripted_model = torch.jit.script(model.cpu())
torch.jit.save(scripted_model, "scripted_model.pth")

# Load the model
try:
    # Option 1: Load state_dict
    loaded_model = ConvNet()
    loaded_model.load_state_dict(torch.load("model_state_dict.pth"))
    loaded_model.eval()
    
    # Option 2: Load scripted model
    loaded_scripted_model = torch.jit.load("scripted_model.pth")
    loaded_scripted_model.eval()
    
    # Test the loaded model
    x = torch.randn((1, 3, 32, 32))
    out = loaded_model(x)
    print("Loaded model output shape:", out.shape)
    
    out_scripted = loaded_scripted_model(x)
    print("Scripted model output shape:", out_scripted.shape)
    
except Exception as e:
    print(f"Error loading model: {e}")
