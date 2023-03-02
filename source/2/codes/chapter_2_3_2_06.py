from torch import nn


class Generator(nn.Module):
    """
    生成器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 200),
            nn.Sigmoid(),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)
