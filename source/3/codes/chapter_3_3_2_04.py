import torch
from torch import nn

from chapter_3_3_2_02 import CommonBlock
from chapter_3_3_2_03 import SpecialBlock


class ResNet18(nn.Module):
    def __init__(self, classes_num=6):
        super(ResNet18, self).__init__()
        # 池化后 —> [batch, 64, 56, 56]
        self.prepare = nn.Sequential(
            # 预卷积操作参数设置
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            # 预卷积操作后—> [batch, 64, 112, 112]
            nn.ReLU(inplace=True),
            # 最大池化参数设置
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            # 第一个残差单元，—> [batch, 64, 56, 56]
            CommonBlock(64, 64, 1),
            # 第二个残差单元，—> [batch, 64, 56, 56]
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            # 第三个残差单元，—> [batch, 128, 28, 28]
            SpecialBlock(64, 128, [2, 1]),
            # 第四个残差单元，—> [batch, 128, 28, 28]
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            # 第五个残差单元，—> [batch, 256, 14, 14]
            SpecialBlock(128, 256, [2, 1]),
            # 第六个残差单元，—> [batch, 256, 14, 14]
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            # 第七个残差单元，—> [batch, 512, 7, 7]
            SpecialBlock(256, 512, [2, 1]),
            # 第八个残差单元，—> [batch, 512, 7, 7]
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # 通过一个自适应均值池化—> [batch, 512, 1, 1]
        self.fc = nn.Sequential(
            # 全连接层，512—>256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 六分类，256—> classes_num == 6
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        """
        使用ResNet18对输入x进行处理，输入x—> [batch, 3, 224, 224]

        Args:
            x:

        Returns:

        """
        x = self.prepare(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        # 返回网络输出结果—>[batch, 6]
        return x


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = ResNet18()
model = model.to(device)
