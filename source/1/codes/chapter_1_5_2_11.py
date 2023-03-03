import numpy as np

# 生成3x3矩阵，其数据值全为0
arr21 = np.zeros([3, 3])
print(arr21)
# 与arr21形状一样的矩阵，其数据值全为0
arr22 = np.zeros_like(arr21)
print(arr22)
# 对角线上元素为1，其他元素全为0的3阶单位矩阵
arr23 = np.eye(3)
print(arr23)
# 对角线上元素为1、2、3，其他全为0的3阶对角矩阵
arr25 = np.diag([1, 2, 3])
print(arr25)
