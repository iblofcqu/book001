from torch import nn
from chapter_2_3_4_01 import Discriminator as Gan


class Discriminator(Gan):
    """
    此时的模型,只是修改了forward,网络主干部分未修改,需注意
    """

    def __init__(self):
        nn.Module.__init__(self)
        self.model = nn.Sequential(
            # 考虑了标签张量的影响
            nn.Linear(784 + 10, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 1),
            nn.Sigmoid())

