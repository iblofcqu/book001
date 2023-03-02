from torch import nn


class Discriminator(nn.Module):
    """
    判别器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)
