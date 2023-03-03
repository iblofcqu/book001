import torch
import numpy as np


def generate_random_one_hot(size):
    label_tensor = torch.zeros(size)
    random_idx = np.random.randint(0, size)
    # 随机令一位为1
    label_tensor[random_idx] = 1
    return label_tensor
