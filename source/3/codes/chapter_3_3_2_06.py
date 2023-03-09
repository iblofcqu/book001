import torch
from torch import nn, optim

from chapter_3_3_2_01 import model, device
from chapter_3_3_1_02 import label_weight
from chapter_3_3_2_05 import train_model

loss_weight = torch.FloatTensor(label_weight).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weight)
# 定义优化函数，adam优化函数
optimizer = optim.Adam(model.parameters(), lr=1e-2)
# 调整学习率，40个epoch学习率衰减0.1
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=40,
                                             gamma=0.9)

# 调用训练函数，开始训练
model_ft = train_model(model,
                       criterion,
                       optimizer,
                       exp_lr_scheduler,
                       num_epochs=1)
