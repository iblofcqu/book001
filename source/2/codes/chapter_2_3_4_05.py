from torch import nn

from chapter_2_3_4_04 import Generator as Gen


class Generator(Gen):

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )
