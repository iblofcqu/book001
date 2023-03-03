import numpy as np
# 导入绘图包
import matplotlib.pyplot as plt

# 在[0,9)范围内以步长为1生成一串数据
arrx = np.arange(0, 9, 1)
# 在[0,4.5)范围内以步长为0.5生成一串数据
arry = np.arange(0, 4.5, 0.5)
# 设置X轴标签，中文需设置字体
plt.xlabel('X轴', fontproperties="SimSun")
# 设置Y轴标签，中文需设置字体
plt.ylabel('Y轴', fontproperties="SimSun")
# 传入需显示图形的x/y轴数据
plt.plot(arrx, arry)
# 显示出图形
plt.show()
