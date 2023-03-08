from torch import nn, functional as F


class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        # 旁支卷积，负责改变输入x维度
        self.change_channel = nn.Sequential(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size=1,
                      stride=stride[0],
                      bias=False),
            nn.BatchNorm2d(out_channel)
        )
        # 第一次卷积操作参数设置
        self.conv1 = nn.Conv2d(in_channel,
                               out_channel,
                               kernel_size=3,
                               stride=stride[0],
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 第二次卷积操作参数设置
        self.conv2 = nn.Conv2d(out_channel,
                               out_channel,
                               kernel_size=3,
                               stride=stride[1],
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        """
        # 调用内部功能函数，对输入x进行处理
        Args:
            x:

        Returns:

        """
        # 输入x经旁支卷积处理后赋给identity
        identity = self.change_channel(x)
        # 对输入x进行第一次卷积操作并激活
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # 第二次卷积操作
        x = self.bn2(self.conv2(x))
        # 将第二次卷积操作的输出x与经旁支卷积处理后的identity相加
        x += identity
        # 激活后返回输出结果
        return F.relu(x, inplace=True)
