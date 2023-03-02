from torch import nn
from chapter_2_3_3_03 import View


class Generator(nn.Module):
    """
    CNN 实现的生成器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # 输入为长度为100的随机生成的噪声
            nn.Linear(100, 32 * 5 * 5),
            nn.LeakyReLU(0.2),
            # 将映射到高维度的特征reshape为32 * 5 * 5的特征图
            View((1, 32, 5, 5)),
            # 第一个转置卷积层，卷积核大小为3 * 3
            nn.ConvTranspose2d(32, 10, kernel_size=3, stride=2),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
            # 第二个转置卷积层，卷积核大小为5 * 5
            nn.ConvTranspose2d(10, 10, kernel_size=5, stride=2,
                               output_padding=1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2),
            # 第三个转置卷积层，卷积核大小为5 * 5
            nn.ConvTranspose2d(10, 1, kernel_size=5, padding=1),
            # 采用Sigmoid作为最后的激活函数，读者可以采用其他函数尝试、对比效果
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)
