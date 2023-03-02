def read_from_csv(csv_path):
    """
    从csv 读取数据的方法，包括测试集和训练集
    :param csv_path: csv 文件的路径
    :return:
    """
    with open(csv_path, "r") as csv_f:
        data_out = csv_f.readlines()
    return data_out


if __name__ == '__main__':
    data_list = read_from_csv("../data/mnist_train.csv")
    print(data_list[0])
