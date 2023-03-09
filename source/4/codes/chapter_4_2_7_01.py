from torch import nn

from chapter_4_2_3_01 import device
from chapter_4_2_5_01 import Generator
from chapter_4_2_6_01 import Discriminator


def weights_init(m):
    """
    模型权重初始化
    Args:
        m:

    Returns:

    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG = Generator().to(device)
netD = Discriminator().to(device)
netG.apply(weights_init)
netD.apply(weights_init)
