import matplotlib.pyplot as plt

from chapter_2_3_4_06 import *


def plot_conditional_images(label):
    label_tensor = torch.zeros(10)
    label_tensor[label] = 1.0
    f, ax_arr = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(2):
        for j in range(3):
            output = generator_net(generate_random(100).to(DEVICE),
                                   label_tensor.to(DEVICE))
            img = output.detach().cpu().numpy().reshape(28, 28)
            ax_arr[i, j].imshow(img, interpolation='None', cmap='Blues')
    plt.show()


plot_conditional_images(9)
