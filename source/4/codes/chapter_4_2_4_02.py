import numpy as np
from matplotlib import pyplot as plt
from torchvision import utils as v_utils

from chapter_4_2_3_01 import device
from chapter_4_2_4_01 import dataloader

real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
img_show = v_utils.make_grid(real_batch[0].to(device)[:64],
                             padding=2,
                             normalize=True)

plt.imshow(np.transpose(img_show.cpu(), (1, 2, 0)))
plt.show()
