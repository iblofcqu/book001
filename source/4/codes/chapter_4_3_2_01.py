from torch import optim

from chapter_4_2_3_01 import beta
from chapter_4_2_7_01 import netG, netD

# 判别器的学习率
lr_D = 0.0003
# 生成器的学习率
lr_G = 0.0001
optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(beta, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(beta, 0.999))
