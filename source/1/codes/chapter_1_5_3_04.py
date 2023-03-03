import numpy as np
import matplotlib.pyplot as plt

import matplotlib.image as img

img = np.random.randint(0, 400, 100)
img = img.reshape(10, 10)
fig, ax = plt.subplots()
# 面向对象方式绘图，fig代表画布，ax代表画布上可绘图区域
im = ax.imshow(img, cmap="seismic")
fig.colorbar(im, orientation="horizontal")
plt.show()
