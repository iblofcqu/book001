import torch

# 创建元素全为0的5×6二维Tensor
xx = torch.zeros(5, 6)
# 创建元素全为1的5×6二维Tensor
yy = torch.ones(5, 6)

print("xx+yy=", xx + yy)
print("xx-yy=", xx - yy)
print("xx*yy=", xx * yy)
print("xx/yy=", xx / yy)
