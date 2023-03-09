from torch import nn
from chapter_4_2_3_01 import ngf, nz, nc


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 100 x 1 x 1
            nn.ConvTranspose2d(in_channels=nz, out_channels=ngf * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4，输入—输出计算过程：O=(I-1)*S+K-2P=(1-1)*1+4-2*0=4
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8，输入—输出计算过程：O=(I-1)*S+K-2P=(4-1)*2+4-2*1=8
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16，输入—输出计算过程：O=(I-1)*S+K-2P=(8-1)*2+4-2*1=16
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32，输入—输出计算过程：O=(I-1)*S+K-2P=(16-1)*2+4-2*1=32
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # (nc) x 64 x 64，输入—输出计算过程：O=(I-1)*S+K-2P=(32-1)*2+4-2*1=64
        )

    def forward(self, inputs):
        return self.main(inputs)
