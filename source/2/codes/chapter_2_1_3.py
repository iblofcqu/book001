"""
章节 2.1.3 顶层调用逻辑
"""
from chapter_2_1_3_10 import test_net
from chapter_2_1_3_06 import make_network
from chapter_2_1_3_01 import read_from_csv
from chapter_2_1_3_11 import count_accuracy
from chapter_2_1_3_08 import single_epoch_train
from chapter_2_1_3_02 import show_csv_line_by_image

# 构建网络
network = make_network()
train_datas = read_from_csv("../data/mnist_train.csv")
show_csv_line_by_image(train_datas[0])
# 训练网络
single_epoch_train(train_datas, network)
# 测试显示网络
test_net(network)
# 计算积分
test_datas = read_from_csv("../data/mnist_test.csv")
count_accuracy(network, test_datas)
