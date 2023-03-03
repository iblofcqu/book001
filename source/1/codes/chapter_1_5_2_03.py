import numpy as np

# 生成3个随机数组成的数组
arr3 = np.random.random(size=3)
# 指定相同的随机数种子，会生成相同的一批随机数
np.random.seed(166)
arr4 = np.random.random(size=3)
print(arr4)
# 随机打乱数组中各元素的顺序
np.random.shuffle(arr4)
print(arr4)
