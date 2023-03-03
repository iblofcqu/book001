import numpy as np

# 2行2列矩阵
arr31 = np.array([[1, 2], [3, 4]])
# 1行2列矩阵
arr32 = np.array([10, 20])
print(arr31)
# arr32会扩展成2行2列，再与arr31相乘
print(arr31 * arr32)
