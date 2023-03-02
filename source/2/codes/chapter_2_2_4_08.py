import os

import numpy as np
import matplotlib.pyplot as plt

from chapter_2_2_4_07 import *


def plot_progress(data, interval):
    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(1, len(data) + 1), data, label='loss')

    plt.xticks(np.arange(0, len(data) + 1, len(data) / 5),
               np.arange(0, len(data) + 1, len(data) / 5,
                         dtype=int) * interval)
    plt.legend()
    plt.savefig(os.path.join("..",
                             "..",
                             "_static",
                             "2",
                             "2.2",
                             "2-48.png"))
    plt.show()


plot_progress(progress, 500)
