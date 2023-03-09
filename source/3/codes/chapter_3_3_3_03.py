import os
import datetime

import torch
from torch.utils import tensorboard

from chapter_3_3_2_01 import device
from chapter_3_3_1_05 import dataloaders
from chapter_3_3_1_02 import dataset_sizes

from chapter_3_3_3_02 import EarlyStopping


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    if not os.path.exists("model"):
        os.mkdir("model")

    model_save_path = os.path.join("model", "best.pt")

    # 创建"logs"文件夹，并以"训练开始日期-时间"为子文件名存储训练数据
    time_path = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tensorboard.SummaryWriter(os.path.join('logs', time_path))
    early_stopping = EarlyStopping(20)
    # 初始化最优准确率
    best_acc = 0.0
    for epoch in range(num_epochs):
        # 记录开始时间
        since = datetime.datetime.now()
        # 存储损失值
        loss_both = {
        }
        # 存储准确率
        acc_both = {
        }
        # 每一个epoch都包含训练集和测试集
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            # 初始化损失值
            running_loss = 0.0
            # 初始化准确率
            running_corrects = 0

            # 开始循环训练，每次从dataloaders读取bach_size个图片和标签。
            for loop_i, datas in enumerate(dataloaders[phase]):
                inputs, labels = datas
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 初始化优化梯度
                optimizer.zero_grad()
                # 训练模式进行如下操作
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # 最后输出的6个结果为六个类别的概率，取最大概率的位置索引赋给preds
                    # 计算输出与标签的损失
                    loss = criterion(outputs, labels)
                    # 打印每个bach_size损失值
                    print(f"{phase}:{loop_i},loss:{loss}")
                    # 训练模式下需要进行反向传播和参数优化
                    if phase == 'train':
                        # 训练模式下计算损失
                        loss.backward()
                        # 训练模式下参数优化方法
                        optimizer.step()
                        # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            # 计算一个epoch损失值
            epoch_loss = running_loss / dataset_sizes[phase]
            # 计算一个epoch准确率
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 将每个epoch损失值存入字典
            loss_both[phase] = epoch_loss
            # 将每个epoch准确率存入字典
            acc_both[phase] = epoch_acc
        # 调整学习率
        scheduler.step()
        # 计算一个epoch时间
        time_elapsed = datetime.datetime.now() - since
        print(
            f"time :{time_elapsed}, epoch :{epoch + 1}, "
            f"loss: {loss_both['train']}, acc :{acc_both['train']}"
            f"val loss:{loss_both['val']},val acc: {acc_both['val']}"
        )
        # 训练完一个epoch后打印： time :xx, epoch :x, loss: xx,
        # acc :xx val loss:xx, val acc: xx
        if acc_both["val"] > best_acc:
            best_acc = acc_both["val"]
            torch.save(model.state_dict(), model_save_path)
        # 将当前epoch的训练结果与过去最好的结果进行比较，如果更好，则在对应地址下更新参数
        # 如果没有变好，则不保存参数。

        # 写入tensorboard 供查看训练过程
        writer.add_scalars("epoch_accuracy", tag_scalar_dict=acc_both,
                           global_step=epoch)
        writer.add_scalars("epoch_loss", tag_scalar_dict=loss_both,
                           global_step=epoch)

        early_stopping(loss_both['val'], model)
        # 判断是否满足停止条件
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # 将训练的参数载入模型
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
    model.eval()
    # 返回带训练参数的模型
    return model
