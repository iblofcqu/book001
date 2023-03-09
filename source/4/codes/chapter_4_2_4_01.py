import os

import torchvision
from torchvision import transforms

from chapter_4_2_3_01 import *

# 数据存放路径
data_path = os.path.join("..", "data", "dataset")
# 创建数据集dataset
dataset = torchvision.datasets.ImageFolder(root=data_path,
                                           transform=transforms.Compose([
                                               transforms.Resize(image_size),
                                               transforms.CenterCrop(
                                                   image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(
                                                   (0.5, 0.5, 0.5),
                                                   (0.5, 0.5, 0.5)),
                                           ]))
# 创建dataloader
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         num_workers=workers)
