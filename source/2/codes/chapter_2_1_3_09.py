import numpy as np

from chapter_2_1_3_06 import NeuralNetwork


def query(network: NeuralNetwork, inputs_list):
    """
    查询神经网络：接受神经网络的输入，返回神经网络的输出
    :param network:
    :param inputs_list:
    :return:
    """
    # 将输入列表转换成二维数组
    inputs = np.array(inputs_list, ndmin=2).T
    # 将输入信号计算到隐藏层
    hidden_inputs = np.dot(network.wih, inputs)
    # 将信号从隐藏层输出
    hidden_outputs = network.activation_function(hidden_inputs)
    # 将信号引入到输出层
    final_inputs = np.dot(network.who, hidden_outputs)
    # 将信号从输出层输出
    final_outputs = network.activation_function(final_inputs)
    # 返回输出层的输出值
    return final_outputs
