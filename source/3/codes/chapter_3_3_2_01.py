import torch
from torch import nn
from torchvision import models

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# 使用预训练模型,随着torchvision 版本变动,该函数实际使用方式在发生变化
model = models.resnet18(pretrained=True)
# 仅取卷积层
fc_inputs = model.fc.in_features
# 载入的resnet18不包含后续的全连接层，需要根据自己项目需求写入
model.fc = nn.Sequential(
    # 全连接层，接256个神经元
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    # 全连接层，接输出
    nn.Linear(256, 6)
)
model = model.to(device)
