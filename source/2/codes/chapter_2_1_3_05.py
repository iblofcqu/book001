import numpy as np
import scipy.special


class NeuralNetwork:
    def __init__(self, input_nodes_num, hidden_nodes_num, output_nodes_num,
                 learning_rate):
        """
        初始化神经网络,构造函数
        :param input_nodes_num:
        :param hidden_nodes_num:
        :param output_nodes_num:
        :param learning_rate:
        """
        # 设置每个输入、隐藏、输出层中的节点数（三层的神经网络）
        self.in_nodes_n = input_nodes_num
        self.hidden_nodes_n = hidden_nodes_num
        self.out_nodes_n = output_nodes_num
        # 权重矩阵， wih代表W input_hidden矩阵，who 代表W hidden_output矩阵
        self.wih = np.random.normal(scale=pow(self.in_nodes_n, -0.5),
                                    size=(self.hidden_nodes_n,
                                          self.in_nodes_n))
        self.who = np.random.normal(scale=pow(self.hidden_nodes_n, -0.5),
                                    size=(self.out_nodes_n,
                                          self.hidden_nodes_n))
        # 学习率
        self.lr = learning_rate
        # 激活函数
        self.activation_function = scipy.special.expit
