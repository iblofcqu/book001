from torch import nn


class Classifier(nn.Module):
    """
    CNN 实现的手写数字分类器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)
