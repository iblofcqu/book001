import numpy as np
import matplotlib.pyplot as plt

data1 = 3 * np.random.random((10, 10))
data2 = 5 * np.random.random((10, 10))
fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
ax2[0].pcolor(data2)
ax2[1].contour(data1)
ax2[2].contourf(data1)
fig2.tight_layout()
plt.show()
