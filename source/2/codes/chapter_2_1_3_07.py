import numpy as np

from chapter_2_1_3_06 import NeuralNetwork


def train(network: NeuralNetwork, inputs_list, targets_list):
    """
    单组数据训练,对应batch size 的x 和 y 数据集
    :param network: 实例化后的网络
    :param inputs_list:
    :param targets_list:
    :return:
    """
    # 将输入列表转换成二维数组
    inputs = np.array(inputs_list, ndmin=2).T
    targets = np.array(targets_list, ndmin=2).T
    # 将输入信号传递到隐藏层
    hidden_inputs = np.dot(network.wih, inputs)
    # 应用激活函数得到隐藏层的输出信号
    hidden_outputs = network.activation_function(hidden_inputs)
    # 将传输的信号传递到输出层
    final_inputs = np.dot(network.who, hidden_outputs)
    # 应用激活函数得到输出层的输出信号
    final_outputs = network.activation_function(final_inputs)
    # 计算输出层的误差：预期目标输出值-网络得到的输出值
    output_errors = targets - final_outputs
    # 计算隐藏层的误差
    hidden_errors = np.dot(network.who.T, output_errors)
    # 反向传播，更新各层权重
    # 更新隐藏层和输出层之间的权重
    network.who += network.lr * np.dot(
        (output_errors * final_outputs * (1.0 - final_outputs)),
        np.transpose(hidden_outputs))
    # 更新输入层和隐藏层之间的权重
    network.wih += network.lr * np.dot(
        (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
        np.transpose(inputs))
