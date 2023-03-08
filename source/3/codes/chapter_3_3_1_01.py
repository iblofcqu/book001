import os

# 读取数据路径与制作标签>
IMAGE_DATASET_PATH = os.path.join(
    "..", "data", "datasets"
)
# 存储标签名
class_name_s = []
# 存储图片地址和标签
data_set = []

for parent_class_name in os.listdir(IMAGE_DATASET_PATH):
    # 返回(路径)下所有文件夹名, ['Decks', 'Pavements', 'Walls']
    for sub_class_name in os.listdir(os.path.join(IMAGE_DATASET_PATH,
                                                  parent_class_name)):
        # 拼接路径，例如：datasets\Decks返回路径下所有文件夹名
        class_name = ".".join([parent_class_name, sub_class_name])
        # 返回parent_class_name.sub_class_name作为标签名

        # 返回class_name_s列表长度
        class_i = len(class_name_s)
        # 将标签名存入class_name_s
        class_name_s.append(class_name)
        # 存储图片路径
        data_x = []
        # 存储图片标签
        data_y = []
        for sub_data_file_name in os.listdir(
                os.path.join(IMAGE_DATASET_PATH,
                             parent_class_name,
                             sub_class_name)):
            # 遍历图片路径并存储在data_x
            data_x.append(os.path.join(IMAGE_DATASET_PATH,
                                       parent_class_name,
                                       sub_class_name,
                                       sub_data_file_name))
            # 将标签存储在data_y
            data_y.append(class_i)
        # 整合图片路径列表和标签列表
        data_set.append((data_x, data_y))
