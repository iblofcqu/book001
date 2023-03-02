import numpy as np

from chapter_2_1_3_09 import query
from chapter_2_1_3_07 import NeuralNetwork
from chapter_2_1_3_03 import scale_inputs_to_01


def count_accuracy(network: NeuralNetwork, datas):
    """
    统计正确率
    :param datas:
    :param network:
    :return:
    """

    # 记录正确的个数
    scorecard_total = 0
    # 遍历测试集中的所有数据
    for record in datas:
        all_values = record.split(',')
        correct_label = int(all_values[0])  # 正确标签
        inputs = scale_inputs_to_01(all_values[1:])
        predict = query(network, inputs)
        predict_label = np.argmax(predict)
        if predict_label == correct_label:
            scorecard_total += 1

    print('Accuracy: ', scorecard_total / len(datas))
