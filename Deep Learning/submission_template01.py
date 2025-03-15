import numpy as np
import torch
from torch import nn

def create_model():
     # Создаем модель с помощью Sequential
    model = nn.Sequential(
        nn.Linear(784, 256, bias=True),  # Первый линейный слой
        nn.ReLU(),                        # Функция активации ReLU
        nn.Linear(256, 16, bias=True),    # Второй линейный слой
        nn.ReLU(),                        # Функция активации ReLU
        nn.Linear(16, 10, bias=True)      # Третий линейный слой (без активации)
    )
    return None

def count_parameters(model):
    # your code here
    # return integer number (None is just a placeholder)
    
    return sum(p.numel() for p in model.parameters())
small_model = nn.Linear(128, 256)
assert count_parameters(small_model) == 128 * 256 + 256, 'Что-то не так, количество параметров неверное'

medium_model = nn.Sequential(*[nn.Linear(128, 32, bias=False), nn.ReLU(), nn.Linear(32, 10, bias=False)])
assert count_parameters(medium_model) == 128 * 32 + 32 * 10, 'Что-то не так, количество параметров неверное'
print("Seems fine!")
