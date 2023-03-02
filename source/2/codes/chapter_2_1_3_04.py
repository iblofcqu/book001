import numpy as np


def make_target(out_nodes_n, label: int):
    """
    构建目标矩阵——onehot标签数据

    :param out_nodes_n: onehot 数据的维度数量
    :param label: index
    :return:
    """
    # 创建用零填充的数组 然后对每个元素加上0.01
    targets = np.zeros(out_nodes_n) + 0.01
    # 手写数字预期值对应的数组元素为0.99
    targets[label] = 0.99
    return targets


if __name__ == '__main__':
    # 将输出节点的数量设置为10
    node_num = 10
    print(make_target(node_num, 7))
