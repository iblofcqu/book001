import os
import numpy as np

arr21 = np.random.random([10, ])
print(arr21)
fileName = os.path.join("..", "data", 'testData.txt')
# 将数组中的数据保存到文件中
np.savetxt(X=arr21, fname=fileName)
# 从文件中恢复数组, arr22与arr21的数据应相同
arr22 = np.loadtxt(fileName)
print(arr22)
