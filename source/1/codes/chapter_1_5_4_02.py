import torch
import numpy as np

# 创建一维Tensor
tsr1 = torch.tensor(np.arange(1, 10, 1))
print(tsr1)
# 创建指定形状的Tensor
tsr2 = torch.tensor((2, 3))
print("tsr2 size:", tsr2.size())
# 创建元素全为0的5×6二维Tensor
xx = torch.zeros(5, 6)
print("xx size", xx.size())
# 创建空（杂乱数据值）的5×6二维Tensor
yy = torch.empty(5, 6)
print("yy size", yy.size())
# 创建随机数组成的5×6二维Tensor，可指定生成方式和值范围
zz = torch.rand(5, 6)

print("zz size", zz.size())
