from chapter_2_1_3_07 import train
from chapter_2_1_3_04 import make_target
from chapter_2_1_3_05 import NeuralNetwork
from chapter_2_1_3_03 import scale_inputs_to_01


def single_epoch_train(datas, network: NeuralNetwork):
    """
    单个epoch 训练逻辑
    :return:
    """
    i = 0
    for record in datas:
        all_values = record.split(',')
        inputs = scale_inputs_to_01(all_values[1:])
        targets = make_target(network.out_nodes_n,
                              int(all_values[0]))
        # 训练网络
        train(network, inputs, targets)
        print('processed data no.', i + 1)
        i += 1
