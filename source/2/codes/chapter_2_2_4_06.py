import torch
from torch import nn

from chapter_2_2_4_05 import Classifier

# 实例化模型
classifier_network = Classifier()
# 损失函数
loss_function = nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(classifier_network.parameters(), lr=0.01)
# 迭代次数计数器
counter = 0
# 记录loss
progress = []
