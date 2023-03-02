import torch

from chapter_2_3_3_01 import Discriminator
from chapter_2_3_2_02 import mnist_dataset
from chapter_2_3_2_09 import generate_random

# 训练CNN GAN判别器
disc_net = Discriminator()
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(disc_net.parameters())
progress = []
counter = 0

for label, image_data_tensor, target in mnist_dataset:
    # 用真实数据、标签1.0训练判别器
    output = disc_net(image_data_tensor.view(1, 1, 28, 28))[0]
    real_loss = loss_function(output, torch.FloatTensor([1.0]))
    optimizer.zero_grad()
    real_loss.backward()
    optimizer.step()

    # 用随机数据、标签0.0训练判别器
    output = disc_net(generate_random((1, 1, 28, 28)))[0]
    fake_loss = loss_function(output, torch.FloatTensor([0.0]))
    optimizer.zero_grad()
    fake_loss.backward()
    optimizer.step()

    counter += 1
    if counter % 500 == 0:
        progress.append(real_loss.item() + fake_loss.item())
    if counter % 10000 == 0:
        print('counter = ', counter)
