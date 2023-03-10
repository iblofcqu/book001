import torch

from chapter_4_2_3_01 import nz, device

# 固定噪声
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
