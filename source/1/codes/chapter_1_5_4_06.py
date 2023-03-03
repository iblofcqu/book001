import torch
import numpy as np

xx = torch.from_numpy(np.arange(0, 30, 1)).reshape(5, 6)
print(xx[0:2, :])
print(xx[:, 1:3])
print(xx[0:2, 1:3])
