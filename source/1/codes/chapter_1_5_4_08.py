import torch
import numpy as np

tsrA1 = torch.from_numpy(np.arange(1, 5, 1))
tsrA2 = torch.from_numpy(np.arange(1, 5, 1).reshape(4, 1))
tsrA3 = torch.from_numpy(np.arange(3, 7, 1))
print(torch.add(tsrA1, tsrA3))
print(torch.addcmul(tsrA1, tsrA2, tsrA3, value=10.0))
print(torch.clamp(tsrA1, 0, 3))
