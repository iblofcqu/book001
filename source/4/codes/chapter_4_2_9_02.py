import os.path

import torch
import torchvision.utils as vutils

from chapter_4_2_7_01 import netG, netD
from chapter_4_2_4_01 import dataloader
from chapter_4_2_9_01 import fixed_noise
from chapter_4_2_3_01 import nz, device, epochs
from chapter_4_2_8_01 import criterion, optimizerD, optimizerG

# 模型训练
real_label = 1.  # 真标签
fake_label = 0.  # 假标签

img_list = []  # 用于存储显示训练过程中生成的图像
G_losses = []  # 用于绘制生成器损失曲线
D_losses = []  # 用于绘制判别器损失曲线
weights_path = "weights"
if not os.path.exists():
    os.mkdir(weights_path)
    
print("开始训练")
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):

        # 用真实数据、real_label训练判别器
        netD.zero_grad()
        # 取出的数据，维度为(128,3,64,64)
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # 真标签，维度为(128,)
        label = torch.full((b_size,), real_label, dtype=torch.float,
                           device=device)
        # 判别器输出，维度为(128,)
        output = netD(real_cpu).view(-1)
        # 基于真实数据，计算判别器损失
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 用噪声生成的数据、fake_label训练判别器
        # 生成一个批次的噪声，维度为(128,100,1,1)
        noise = torch.randn(b_size, nz, 1, 1,
                            device=device)
        # 噪声输入生成器，得到假图片，维度为(128,3,64,64)
        fake = netG(noise)
        # 假标签，维度为(128,)
        label.fill_(fake_label)
        # 判别器输出，维度为(128,)
        output = netD(fake.detach()).view(-1)
        # 基于生成数据，计算判别器损失
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 总的判别器损失
        errD = errD_real + errD_fake
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        # 计算生成器损失
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # 显示训练过程中的模型损失和判别值
        if i % 10 == 0:
            print(
                '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch + 1, epochs, i, len(dataloader),
                   errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # 保存生成器在训练过程中生成的图像
        if ((epoch + 1) % 10 == 0 or epoch == 0) and i == len(dataloader) - 1:
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(
                vutils.make_grid(fake, nrow=6, padding=3, normalize=True))

        # 保存生成器和判别器的参数
        if (epoch + 1) % 100 == 0:
            torch.save(netG.state_dict(),
                       './weights/netG_epoch_{}.pth'.format(epoch + 1))
            torch.save(netD.state_dict(),
                       './weights/netD_epoch_{}.pth'.format(epoch + 1))
