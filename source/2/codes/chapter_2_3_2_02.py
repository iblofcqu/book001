import matplotlib.pyplot as plt

from chapter_2_2_4_03 import MnistDataset as Dataset


class MnistDataset(Dataset):
    """
    通过继承,复用2.2 章节中重复的代码
    """

    def plot_image(self, index):
        img = self.data_df.iloc[index, 1:].values.reshape(28, 28)
        plt.title("label = " + str(self.data_df.iloc[index, 0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        plt.show()


mnist_dataset = MnistDataset('../data/mnist_train.csv')
if __name__ == '__main__':
    mnist_dataset.plot_image(0)
