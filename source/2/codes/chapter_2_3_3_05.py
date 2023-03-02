import torch
import matplotlib.pyplot as plt

from chapter_2_3_2_02 import mnist_dataset
from chapter_2_3_3_04 import Generator
from chapter_2_3_3_01 import Discriminator
from chapter_2_3_2_09 import generate_random

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_image():
    f, ax_arr = plt.subplots(2, 3, figsize=(16, 8))
    for _i in range(2):
        for j in range(3):
            outputs = generator_net(generate_random(100).to(DEVICE))
            img = outputs.detach().cpu().numpy().reshape(28, 28)
            ax_arr[_i, j].imshow(img, interpolation='None', cmap='Blues')
    plt.show()


discriminator_net = Discriminator().to(DEVICE)
generator_net = Generator().to(DEVICE)
loss_function = torch.nn.BCELoss()

optimizer_d = torch.optim.Adam(discriminator_net.parameters())
optimizer_g = torch.optim.Adam(generator_net.parameters())
progress_d_real = []
progress_d_fake = []
progress_g = []
counter = 0

for i in range(5):
    for label, real_data, target in mnist_dataset:
        discriminator_net.zero_grad()
        real_label = torch.FloatTensor([1.0]).to(DEVICE)
        fake_label = torch.FloatTensor([0.0]).to(DEVICE)
        img_inputs = real_data.view(1, 1, 28, 28)
        output = discriminator_net(img_inputs.to(DEVICE))[0]
        loss_d_real = loss_function(output, real_label)
        gen_img = generator_net(generate_random(100).to(DEVICE))
        output = discriminator_net(gen_img.detach())[0]
        loss_d_fake = loss_function(output, fake_label)
        loss_d = loss_d_real + loss_d_fake
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()

        generator_net.zero_grad()
        gen_img = generator_net(generate_random(100).to(DEVICE))
        output = discriminator_net(gen_img)[0]
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

    plot_image()
