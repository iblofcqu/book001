import numpy as np
import matplotlib.pyplot as plt

arrx = np.arange(0, 9, 1)
arrY1 = np.arange(0, 4.5, 0.5)
plt.xlabel('X轴', fontproperties="SimSun")
plt.ylabel('Y轴', fontproperties="SimSun")
plt.plot(arrx, arrY1, label="line")
arrx = np.arange(0, 9, 1)
arrY2 = np.sin(arrx)
plt.plot(arrx, arrY2, label="sin")
arrY3 = np.cos(arrx)
plt.title("sin/cos曲线", fontproperties="SimSun")  # 图标题
plt.plot(arrx, arrY3, linestyle="--", label="cos")  # 虚线显示
plt.legend()  # 显示线段说明文本标签
plt.show()  # 显示出图形
