import torch
import torch.nn as nn

def count_parameters(model):
    """
    Подсчитывает количество параметров в модели.

    Args:
        model: Модель PyTorch.

    Returns:
        int: Количество параметров в модели.
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    # верните количество параметров модели model
    return total_params

# не изменяйте код в блоке ниже! Он нужен для проверки правильности вашего кода.
# __________start of block__________
small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Что-то не так, количество параметров неверное'

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Что-то не так, количество параметров неверное'
print("Seems fine!")
# __________end of block__________
