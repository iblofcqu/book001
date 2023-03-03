import numpy as np

arr26 = np.linspace(1, 25, 25).reshape([5, 5])

# 将一维数组转换成5行5列矩阵
print(arr26)
# 在原矩阵中，取[1,4)行与[1,4)列
print(arr26[0:3, 0:3])
# 在原矩阵中，取[2,5)行与所有列
print(arr26[1:4, :])
arr27 = np.arange(1, 25, dtype=float)
print(arr27)
cse1 = np.random.choice(arr27, size=(4, 3))

# 从数组中随机抽取数，并返回4行3列的矩阵
print(cse1)
cse2 = np.random.choice(arr27, size=(4, 3), p=arr27 / np.sum(arr27))

# 同上，但指定概率抽数

print(cse2)
