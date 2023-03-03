import os
import matplotlib.pyplot as plt

import matplotlib.image as img

img_path = os.path.join("..", "..", "_static", "1", "1.5", "1-23.png")
img1 = img.imread(img_path)
plt.imshow(img1)
plt.show()
