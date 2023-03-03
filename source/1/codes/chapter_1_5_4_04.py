import torch

xx = torch.zeros(5, 6)
print("xx size:", xx.size())
yy = xx.reshape(30)
print("yy size:", yy.size())
zz = xx.reshape(5, 3, 2)
print("zz size:", zz.size())
