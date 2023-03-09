import torch
from torch import nn
from torchvision import models

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# 使用预训练模型
model = models.resnet18(pretrained=True)
fc_inputs = model.fc.in_features
# 载入的resnet18不包含后续的全连接层，需要根据自己项目写入
model.fc = nn.Sequential(
    # ——以p=0.5添加项Dropout——
    nn.Dropout(p=0.5),
    # 全连接层，接256个神经元
    nn.Linear(fc_inputs, 256),
    # 激活
    nn.ReLU(),
    # ——以p=0.5添加项Dropout——
    nn.Dropout(p=0.5),
    # 全连接层，接输出
    nn.Linear(256, 6)
)
# 将由CPU保存的模型加载到GPU上，提高训练速度
model = model.to(device)
