from PIL import Image

from torch.utils import data as torch_data

from chapter_3_3_1_03 import data_transforms
from chapter_3_3_1_02 import x_train_s, y_train_s, x_test_s, y_test_s


class CrackDataset(torch_data.Dataset):
    """
    定义一个读取图片信息的类
    """
    _files = None
    _labels = None
    _transform = False

    def __init__(self, abs_file_path_s, y_datas, trans=False):
        """

        Args:
            abs_file_path_s:
            y_datas:
            trans: 该数据集可以使用的数据增强方式集
        """
        self._files = abs_file_path_s  # 传入图像地址
        self._labels = y_datas  # 传入图像标签
        self._transform = trans  # 传入需要载入的预处理函数

    def __getitem__(self, item):
        """
        通过python 魔法方法扩写,使其呈现出列表，按索引取值的功能，时间换空间，避免内存爆炸
        Args:
            item:

        Returns:

        """
        img = Image.open(self._files[item]).convert("RGB")
        # 按顺序读取图片标签
        label = self._labels[item]
        if self._transform:
            # 对图片进行预处理or增强
            img = self._transform(img)
        # 返回预处理后的图像信息和标签
        return img, label

    def __len__(self):
        """
        返回数据长度;同属于python 魔法方法扩写,用于支持len()函数调用
        Returns:

        """
        return len(self._files)


train_data = CrackDataset(abs_file_path_s=x_train_s, y_datas=y_train_s,
                          trans=data_transforms["train"])  # 调用类功能，输出训练数据
test_data = CrackDataset(abs_file_path_s=x_test_s, y_datas=y_test_s,
                         trans=data_transforms["val"])  # 调用类功能，输出测试数据
