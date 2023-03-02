from torch import nn


class Discriminator(nn.Module):
    """
    采用卷积神经网络实现的判别器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)
