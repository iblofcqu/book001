import numpy as np
import matplotlib.pyplot as plt

from chapter_2_1_3_09 import query
from chapter_2_1_3_01 import read_from_csv
from chapter_2_1_3_07 import NeuralNetwork
from chapter_2_1_3_03 import scale_inputs_to_01
from chapter_2_1_3_02 import show_csv_line_by_image


def test_net(network: NeuralNetwork):
    test_datas = read_from_csv("../data/mnist_test.csv")
    # 显示第一个数据
    show_csv_line_by_image(test_datas[0])
    # 将第一条数据传入网络预测
    all_values = test_datas[0].split(',')
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    # 查看标签
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    # 查询经网络的输出
    print(query(network, scale_inputs_to_01(all_values[1:])))


