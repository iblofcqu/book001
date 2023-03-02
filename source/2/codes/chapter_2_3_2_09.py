"""
改良后的GAN 手写生成
"""
import os

import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from chapter_2_3_2_02 import MnistDataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    """
    修改后的判别器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)


class Generator(nn.Module):
    """
    修改后的生成器
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LayerNorm(200),
            nn.LeakyReLU(0.02),
            nn.Linear(200, 784),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.model(inputs)


def generate_random(size):
    random_data = torch.randn(size)
    return random_data


def plot_image(gen_net: Generator):
    f, ax_arr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            outputs = gen_net(generate_random(100).to(DEVICE))
            img = outputs.detach().cpu().numpy().reshape(28, 28)
            ax_arr[i, j].imshow(img, interpolation='None', cmap='Blues')
    plt.show()


def start():
    train_csv_path = os.path.join("..", "data", "mnist_train.csv")
    train_dataset = MnistDataset(train_csv_path)

    # demo 3 modified
    discriminator_net = Discriminator().to(DEVICE)
    generator_net = Generator().to(DEVICE)
    loss_function = torch.nn.BCELoss()

    optimizer_d = torch.optim.Adam(discriminator_net.parameters())
    optimizer_g = torch.optim.Adam(generator_net.parameters())
    progress_d_real = []
    progress_d_fake = []
    progress_g = []
    counter = 0

    real_label = torch.FloatTensor([1.0]).to(DEVICE)
    fake_label = torch.FloatTensor([0.0]).to(DEVICE)

    for i in range(10):
        for label, real_data, target in train_dataset:
            discriminator_net.zero_grad()

            output = discriminator_net(real_data.to(DEVICE))
            loss_d_real = loss_function(output, real_label)

            gen_img = generator_net(generate_random(100).to(DEVICE))
            output = discriminator_net(gen_img.detach())
            loss_d_fake = loss_function(output, fake_label)
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            generator_net.zero_grad()
            img_gen = generator_net(generate_random(100).to(DEVICE))
            output = discriminator_net(img_gen)
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
                print(loss_d.item(), loss_g.item())

        plot_image(generator_net)


if __name__ == '__main__':
    start()
