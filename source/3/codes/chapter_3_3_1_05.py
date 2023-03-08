from torch.utils import data as torch_data

from chapter_3_3_1_04 import train_data, test_data

# 一次输入网络模型的图片量，根据开发环境GPU 大小调整
BATCH_SIZE = 64
train_loader = torch_data.DataLoader(train_data, batch_size=BATCH_SIZE,
                                     shuffle=True)
test_loader = torch_data.DataLoader(test_data, batch_size=BATCH_SIZE)

# 将训练数据和测试数据存入字典dataloaders
dataloaders = {
    'train': train_loader,
    'val': test_loader,
}
