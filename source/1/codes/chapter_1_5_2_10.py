import numpy as np

# 生成2行3列数组（又称矩阵）
arr3 = np.array([[1, 2, 3], [6, 7, 8]])
# 生成2行3列数组（又称矩阵）
arr4 = np.array([[3, 2, 1], [8, 7, 6]])
# 查看+运算结果，可发现是逐个元素相加
print("arr3 + arr4=:", arr3 + arr4)
# 查看*运算结果，可发现是逐个元素相乘
print("arr3 * arr4=:", arr3 * arr4)
# 查看*向量后运算结果，可发现是逐个元素相乘
print("arr3 * 10=:", arr3 * 10)
