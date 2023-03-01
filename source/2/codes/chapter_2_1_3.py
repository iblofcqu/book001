"""
章节 2.1.3 示例代码汇总
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt


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


class NetWorkWithHandle(NeuralNetwork):
    def train(self, inputs_list, targets_list):
        """
        单组数据训练,对应batch size 的x 和 y 数据集
        :param inputs_list:
        :param targets_list:
        :return:
        """
        # 将输入列表转换成二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 将输入信号传递到隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        # 应用激活函数得到隐藏层的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将传输的信号传递到输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 应用激活函数得到输出层的输出信号
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层的误差：预期目标输出值-网络得到的输出值
        output_errors = targets - final_outputs
        # 计算隐藏层的误差
        hidden_errors = np.dot(self.who.T, output_errors)
        # 反向传播，更新各层权重
        # 更新隐藏层和输出层之间的权重
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs))
        # 更新输入层和隐藏层之间的权重
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs))

    # 查询神经网络：接受神经网络的输入，返回神经网络的输出
    def query(self, inputs_list):
        # 将输入列表转换成二维数组
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入信号计算到隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        # 将信号从隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 将信号引入到输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 将信号从输出层输出
        final_outputs = self.activation_function(final_inputs)
        # 返回输出层的输出值
        return final_outputs


def read_from_csv(csv_path):
    """
    从csv 读取数据的方法，包括测试集和训练集
    :param csv_path: csv 文件的路径
    :return:
    """
    with open(csv_path, "r") as csv_f:
        training_data_list = csv_f.readlines()
    return training_data_list


def show_csv_line_by_image(values):
    """
    将csv 数据 [1:] 部分数据转换成图片绘制出来
    :param values: 单行数据,此时仍然包括了原本index 为 0 的标签数据
    :return:
    """
    all_values = values.split(',')
    images_array = np.asfarray(all_values[1:]).reshape(28, 28)  # 转化为矩阵
    plt.imshow(images_array, cmap='Greys', interpolation='None')  # 展示图像
    plt.show()


def make_target(out_nodes_n, label: int):
    """
    构建目标矩阵——onehot标签数据
    :param out_nodes_n: onehot 数据的维度数量
    :param label: index
    :return:
    """
    targets = np.zeros(out_nodes_n) + 0.01
    targets[label] = 0.99
    return targets


class StudyClient:
    """
    封装运行逻辑,便于sphinx include 指令提取代码显示
    """

    training_data_list = None
    test_data_list = None
    network: NetWorkWithHandle
    output_nodes_num: int

    def __init__(self):
        self.training_data_list = read_from_csv("../data/mnist_train.csv")

        self.show_first_train_csv_line_by_plt()
        self.set_network()

    def show_first_train_csv_line_by_plt(self):
        """
        将csv 训练集中第一行数据转换成图片并绘图展示
        :return:
        """
        show_csv_line_by_image(self.training_data_list[0])

    def test_first_test_csv_line(self):
        """
        对测试集中的数据做展示性输出

        :return:
        """
        self.test_data_list = read_from_csv("../data/mnist_test.csv")
        # 获取数据集第一个数据
        show_csv_line_by_image(self.test_data_list[0])
        # 转化数组
        all_values = self.test_data_list[0].split(',')
        image_array = np.asfarray(all_values[1:]).reshape((28, 28))
        # 查看标签
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        # 查询经网络的输出
        print(self.network.query(scale_inputs_to_01(all_values[1:])))

    def set_network(self):
        """
        配置网络

        :return:
        """
        # 输入节点 28*28
        input_nodes_num = 784
        # 隐藏层节点
        hidden_nodes_num = 100
        # 输出层节点
        self.output_nodes_num = 10
        # 设置学习率
        learning_rate = 0.3
        # NeuralNetwork 子类
        self.network = NetWorkWithHandle(input_nodes_num,
                                         hidden_nodes_num,
                                         self.output_nodes_num,
                                         learning_rate)

    def single_epoch_train(self):
        """
        单个epoch 训练逻辑
        :return:
        """
        i = 0
        for record in self.training_data_list:
            all_values = record.split(',')
            inputs = scale_inputs_to_01(all_values[1:])
            targets = make_target(self.output_nodes_num,
                                  int(all_values[0]))
            # 训练网络
            self.network.train(inputs, targets)
            print('processed data no.', i + 1)
            i += 1

    def epoch_i_train(self, epoch: int = 1):
        for _ in range(epoch):
            self.single_epoch_train()

    def count_accuracy(self):
        """
        统计正确率
        :return:
        """
        # 记录正确的个数
        scorecard_total = 0
        # 遍历测试集中的所有数据
        for record in self.test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])  # 正确标签
            inputs = scale_inputs_to_01(all_values[1:])
            predict = self.network.query(inputs)
            predict_label = np.argmax(predict)
            if predict_label == correct_label:
                scorecard_total += 1

        print('Accuracy: ', scorecard_total / len(self.test_data_list))


def scale_inputs_to_01(img_values):
    """
    将csv 中数据集,映射到 0-1
    :param img_values: csv 单行数据[1:]部分,且已经处理成List[int] 形式
    :return: 缩放后的像素点
    """
    scaled = (np.asfarray(img_values) / 255.0 * 0.99) + 0.01
    return scaled


if __name__ == '__main__':
    client = StudyClient()
    # 训练一次
    client.epoch_i_train()

    client.test_first_test_csv_line_by_plt()
    client.count_accuracy()
