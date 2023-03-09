import torch
from torch import nn

from chapter_4_2_3_01 import lr, beta
from chapter_4_2_7_01 import netG, netD

# 损失函数与优化器
criterion = nn.BCELoss()
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta, 0.999))
