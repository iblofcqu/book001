import torch

from chapter_2_3_2_06 import Generator
from chapter_2_3_2_03 import Discriminator
from chapter_2_2_4_03 import train_dataset
from chapter_2_3_2_04 import generate_random

discriminator = Discriminator()
gen = Generator()
loss_function = torch.nn.BCELoss()
optimizer_d = torch.optim.Adam(discriminator.parameters())
optimizer_g = torch.optim.Adam(gen.parameters())
progress_d = []
progress_g = []
epoch_s = 10

for i in range(epoch_s):
    counter = 0
    for label, real_data, target in train_dataset:
        # (1) 用真实数据、1.0训练判别器
        output = discriminator(real_data)
        loss_d_real = loss_function(output, torch.FloatTensor([1.0]))
        optimizer_d.zero_grad()
        loss_d_real.backward()
        optimizer_d.step()
        # (2) 用生成数据、0.0训练判别器
        output = discriminator(gen(generate_random(1)).detach())
        loss_d_fake = loss_function(output, torch.FloatTensor([0.0]))
        optimizer_d.zero_grad()
        loss_d_fake.backward()
        optimizer_d.step()
        # (3) 训练生成器
        output = discriminator(gen(generate_random(1)))
        loss_g = loss_function(output, torch.FloatTensor([0.5]))
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        counter += 1
        # 保存loss，输出训练进度
        if counter % 500 == 0:
            progress_d.append(loss_d_fake.item() + loss_d_real.item())
            progress_g.append(loss_g.item())
        if counter % 10000 == 0:
            print('epoch = {}, counter = {}'.format(i + 1, counter))
