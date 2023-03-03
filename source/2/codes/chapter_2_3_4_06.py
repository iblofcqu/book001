import torch

from chapter_2_3_4_05 import Generator
from chapter_2_3_4_02 import Discriminator
from chapter_2_3_2_02 import mnist_dataset
from chapter_2_3_2_09 import generate_random
from chapter_2_3_4_03 import generate_random_one_hot

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

discriminator_net = Discriminator().to(DEVICE)
generator_net = Generator().to(DEVICE)
loss_function = torch.nn.BCELoss()

optimizer_d = torch.optim.Adam(discriminator_net.parameters())
optimizer_g = torch.optim.Adam(generator_net.parameters())
progress_d_real = []
progress_d_fake = []
progress_g = []
counter = 0
# 真假标签
real_label = torch.FloatTensor([1.0]).to(DEVICE)
fake_label = torch.FloatTensor([0.0]).to(DEVICE)

for i in range(10):
    for label, real_data, target in mnist_dataset:
        discriminator_net.zero_grad()
        # 真实数据训练判别器
        output = discriminator_net(real_data.to(DEVICE), target.to(DEVICE))
        loss_d_real = loss_function(output, real_label)

        # 生成数据训练判别器
        random_label = generate_random_one_hot(10).to(DEVICE)
        gen_img = generator_net(generate_random(100).to(DEVICE),
                                random_label)
        output = discriminator_net(gen_img.detach(), random_label)
        loss_d_fake = loss_function(output, fake_label)
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        # 训练生成器,使生成器生成的图像更真实
        generator_net.zero_grad()
        gen_img = generator_net(generate_random(100).to(DEVICE),
                                random_label)
        output = discriminator_net(gen_img,
                                   random_label)
        loss_g = loss_function(output, real_label)
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        counter += 1
        if counter % 500 == 0:
            progress_d_real.append(loss_d_real.item())
            progress_d_fake.append(loss_d_fake.item())
            progress_g.append(loss_g.item())
        if counter % 10000 == 0:
            print(f'epoch = {i + 1}, counter = {counter}')
