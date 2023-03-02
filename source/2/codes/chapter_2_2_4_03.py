import pandas
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        label = self.data_df.iloc[index, 0]
        target = torch.zeros(10)
        target[label] = 1.0
        image_values = torch.FloatTensor(
            self.data_df.iloc[index, 1:].values) / 255.0
        return label, image_values, target


# 实例化，获得数据集
train_dataset = MnistDataset('../data/mnist_train.csv')
test_dataset = MnistDataset('../data/mnist_test.csv')
