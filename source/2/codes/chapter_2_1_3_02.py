import numpy as np
import matplotlib.pyplot as plt

from chapter_2_1_3_01 import read_from_csv


def show_csv_line_by_image(values):
    """
    将csv 数据 [1:] 部分数据转换成图片绘制出来
    :param values: sting,will split by ,
    :return:
    """
    all_values = values.split(',')
    # 转化为矩阵
    images_array = np.asfarray(all_values[1:]).reshape(28, 28)
    # 展示图像
    plt.imshow(images_array, cmap='Greys', interpolation='None')
    plt.show()


if __name__ == '__main__':
    show_csv_line_by_image(read_from_csv("../data/mnist_train.csv")[0])
