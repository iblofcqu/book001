from chapter_2_1_3_05 import NeuralNetwork


def make_network():
    # 输入节点 28*28
    input_nodes_num = 784
    # 隐藏层节点
    hidden_nodes_num = 100
    # 输出层节点
    output_nodes_num = 10
    # 设置学习率
    learning_rate = 0.3
    # NeuralNetwork 子类
    network = NeuralNetwork(input_nodes_num,
                            hidden_nodes_num,
                            output_nodes_num,
                            learning_rate)
    return network
