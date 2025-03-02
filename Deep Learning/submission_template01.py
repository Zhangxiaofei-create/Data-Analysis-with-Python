import torch
import torch.nn as nn
import re

# импортируйте torch и сохраните его версию в переменную version
# your code here
version = torch.__version__

# не изменяйте код в блоке ниже! Он нужен для проверки правильности вашего кода.
# __________start of block__________
assert version is not None, 'Версия PyTorch не сохранилась в переменную version'
major_version, minor_version = re.findall("\d+\.\d+", version)[0].split('.')
assert float(major_version) >= 2 or (float(major_version) >= 1 and float(minor_version) >= 7), 'Нужно обновить PyTorch'
# __________end of block__________

def create_model():
    """
    Создает нейронную сеть из трех линейных слоев с ReLU в качестве функций активации
    между слоями.
    """
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),
        nn.ReLU(),
        nn.Linear(256, 16, bias=True),
        nn.ReLU(),
        nn.Linear(16, 10, bias=True)
    )
    return model
