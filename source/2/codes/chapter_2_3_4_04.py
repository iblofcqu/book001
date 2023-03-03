import torch

from chapter_2_3_2_09 import Generator as Gen


class Generator(Gen):
    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)
