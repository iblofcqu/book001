from torch import nn

from chapter_4_2_3_01 import nc, ndf


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=nc, out_channels=ndf,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        # 获取第一层输出的特征
        out1 = self.layer1(inputs)
        # 获取第二层输出的特征
        out2 = self.layer2(out1)
        # 获取第三层输出的特征
        out3 = self.layer3(out2)
        # 获取第四层输出的特征
        out4 = self.layer4(out3)
        # 获取模型输出
        out5 = self.layer5(out4)
        return out5, [out1, out2, out3, out4]
