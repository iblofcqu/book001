import torch
from chapter_2_3_2_09 import Discriminator as Gan


class Discriminator(Gan):
    """
    此时的模型,只是修改了forward,网络主干部分未修改,需注意
    """

    def forward(self, image_tensor, label_tensor):
        inputs = torch.cat((image_tensor, label_tensor))
        return self.model(inputs)
