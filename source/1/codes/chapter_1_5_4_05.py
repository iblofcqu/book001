import torch

xx = torch.zeros(5, 6)
yy = xx.numpy()
print("yy type=", type(yy))
zz = torch.from_numpy(yy)
print("zz type=", type(zz))
