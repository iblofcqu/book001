import numpy as np

# 创建一个Python列表
lst1 = [1, 2, 3, 5]
# 调用NumPy函数array，传入列表，生成一个NumPy数组
arr1 = np.array(lst1)
# 查看Python列表内容
print(lst1)
# 查看NumPy数组内容，和上面代码显示的数据内容一样
print(arr1)
# 查看其类型
print(type(lst1))
# 与上行代码比，两者数据相同，但类型不同、存储方式不同
print(type(arr1))
