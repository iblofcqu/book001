from torch import nn
from chapter_4_2_3_01 import nc, ndf


class Discriminator(nn.Module):
    """
    DCGAN判别器的构建
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32，输入—输出计算过程：
            # O=(I-K+2P)/S+1=(64-4+2*1)/2+1=32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16，输入—输出计算过程：
            # O=(I-K+2P)/S+1=(32-4+2*1)/2+1=16
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8，输入—输出计算过程：
            # O=(I-K+2P)/S+1=(16-4+2*1)/2+1=8
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4，输入—输出计算过程：
            # O=(I-K+2P)/S+1=(8-4+2*1)/2+1=4
            nn.Conv2d(in_channels=ndf * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # 1 x 1 x 1，输入—输出计算过程：
            # O=(I-K+2P)/S+1=(4-4+2*0)/2+1=1
        )

    def forward(self, inputs):
        return self.main(inputs)
