import random

import numpy as np
from sklearn import model_selection

from chapter_3_3_1_01 import data_set, class_name_s

# 用来存储训练集图片路径
x_train_s = []
# 用来存储测试集图片路径
x_test_s = []
# 用来存储训练集标签
y_train_s = []
# 用来存储测试集标签
y_test_s = []
for x_data, y_data in data_set:
    # 每个类别的所有图片路径和标签按8：2分为训练集和测试集，训练标签和测试标签
    x_tr, x_te, y_tr, y_te = model_selection.train_test_split(x_data,
                                                              y_data,
                                                              test_size=0.2)
    # 六个类别的训练集路径依次存入x_train_s
    x_train_s.extend(x_tr)
    # 六个类别的测试集路径依次存入x_test_s
    x_test_s.extend(x_te)
    # 六个类别的训练标签依次存入y_train_s
    y_train_s.extend(y_tr)
    # 六个类别的测试标签依次存入y_test_s
    y_test_s.extend(y_te)
# 可以发现，上述划分训练集和测试集时并不是对整个数据按照比例进行划分，
# 而是对每个类别分别进行划分后合并，这样做可以提高每个类别训练样本的均匀性。
# print(len(x_train_s)) = 44871, 训练集的数量约为总数的8/10。

train_data = list(zip(x_train_s, y_train_s))
# 将训练集和训练标签合并—>[('图片地址1', 标签1),……('图片地址44871', 标签44871)]
random.shuffle(train_data)
# 将列表中元素顺序打乱
x_train_s, y_train_s = list(zip(*train_data))
# 将打乱后的训练数据拆分成新的训练集和训练标签
# 对测试集路径做相同的打乱和拆分处理
test_data = list(zip(x_test_s, y_test_s))
random.shuffle(test_data)
x_test_s, y_test_s = list(zip(*test_data))
# 保存训练集和测试集数据量
dataset_sizes = {
    "train": len(train_data),
    "val": len(test_data),
}

# 计算样本权重，可以得到每个样本所占比重，是后续设置损失函数时所需参数
# 将训练集标签转为数组
y_train_np = np.asarray(y_train_s)
# 将训练集标签数组转为独热编码
y_one_hot = np.eye(len(class_name_s))[y_train_np]
# 列向量求和，可知道训练集中各类别的数量
class_count = np.sum(y_one_hot, axis=0)
# 求六个类别总数量—>44871
total_count = np.sum(class_count)
# 求样本权重
label_weight = (1 / class_count) * total_count / 2
