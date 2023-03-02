import torch
from torch import nn

from chapter_2_2_4_03 import train_dataset
from chapter_2_3_2_03 import Discriminator
from chapter_2_3_2_04 import generate_random

# 判别器
discriminator = Discriminator()
# 损失函数
loss_function = nn.MSELoss()
# 优化器
optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01)
counter = 0
progress = []

for label, image_data_tensor, target in train_dataset:
    # 用真实数据、标签1.0训练判别器
    output = discriminator(image_data_tensor)
    real_loss = loss_function(output, torch.FloatTensor([1.0]))

    # 用随机数据、标签0.0训练判别器
    output = discriminator(generate_random(784))
    fake_loss = loss_function(output, torch.FloatTensor([0.0]))

    # 反向传播，更新参数
    loss = real_loss + fake_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    counter += 1
    if counter % 500 == 0:
        progress.append(real_loss.item() + fake_loss.item())
    if counter % 10000 == 0:
        print('counter = ', counter)
