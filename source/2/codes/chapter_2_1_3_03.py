import numpy as np

from chapter_2_1_3_01 import read_from_csv


def scale_inputs_to_01(img_values):
    """
    将csv 中数据集,映射到 0-1
    :param img_values: csv 单行数据[1:]部分,已经被 , 分割
    :return: 缩放后的像素点
    """
    scaled = (np.asfarray(img_values) / 255.0 * 0.99) + 0.01
    return scaled


if __name__ == '__main__':
    train_datas = read_from_csv("../data/mnist_train.csv")
    print(scale_inputs_to_01(train_datas[0].split(',')[1:]))
