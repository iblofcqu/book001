import torch
import numpy as np

arrA = np.arange(0, 50, 10).reshape(5, 1)
arrB = np.arange(0, 3, 1)
tsrA1 = torch.from_numpy(arrA)
print("tsrA1 size:", tsrA1.size())
tsrB1 = torch.from_numpy(arrB)
print("tsrB1 size:", tsrB1.size())
tsrC1 = tsrA1 + tsrB1
print("tsrC1 size:", tsrC1.size())
# 广播的实现过程
tsrB2 = tsrB1.unsqueeze(0)
tsrA2 = tsrA1.expand(5, 3)
tsrB3 = tsrB2.expand(5, 3)
tsrC2 = tsrA2 + tsrB3
print("tsrC2 size:", tsrC2.size(), " equal:", tsrC2 == tsrC1)
