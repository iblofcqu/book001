import torch
from torch import nn, optim

from chapter_4_2_3_01 import epochs, device, nz, beta
from chapter_4_2_4_01 import dataloader
from chapter_4_3_1_01 import D_real_label, D_fake_label
from chapter_4_2_7_01 import netG
from chapter_4_3_2_01 import optimizerG
from chapter_4_3_3_01 import Discriminator

# 使用新的Discriminator
netD = Discriminator().to(device)
optimizerD = optim.Adam(netD.parameters(), lr=0.0003, betas=(beta, 0.999))
# 特征匹配损失
criterion = nn.BCELoss()
criterionG = nn.MSELoss()

for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # 用真实数据、real_label训练判别器
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        # 真标签
        label = torch.full((b_size,), D_real_label, dtype=torch.float,
                           device=device)
        # 使用NULLD变量表示不使用的输出
        output, NULLD = netD(real_cpu)
        output = output.view(-1)
        # 计算判别器对真实数据的损失
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # 用噪声生成的数据、fake_label训练判别器
        # 生成一个批次的噪声
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # 噪声输入生成器，得到生成图片
        fake = netG(noise)
        # 假标签
        label.fill_(D_fake_label)
        # 使用NULLD变量表示不使用的输出
        output, NULLD = netD(fake.detach())
        output = output.view(-1)
        # 计算判别器对生成数据的损失
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 计算判别器的损失
        errD = errD_real + errD_fake
        optimizerD.step()

        # 训练生成器
        netG.zero_grad()
        # 基于真实数据，获取判别器特征
        _, feature_real = netD(real_cpu.detach())
        # 基于生成数据，获取判别器特征
        output, feature_fake = netD(fake)
        # 计算真实数据在判别器最后一层特征的期望
        feature_real_last = torch.mean(feature_real[-1], 0)
        # 计算生成数据在判别器最后一层特征的期望
        feature_fake_last = torch.mean(feature_fake[-1], 0)
        # 使用均方误差计算真实样本与生成样本的特征的损失
        errG = criterionG(feature_fake_last, feature_real_last.detach())
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
