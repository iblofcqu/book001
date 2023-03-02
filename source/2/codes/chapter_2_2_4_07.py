from torch.utils.data import DataLoader

from chapter_2_2_4_03 import train_dataset
from chapter_2_2_4_06 import *

# GPU or CPU 环境判断
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

train_loader = DataLoader(train_dataset, batch_size=16)

classifier_network = classifier_network.to(device)
loss_tmp = 0
for i in range(20):
    for label, image_data_tensor, target in train_loader:
        reshaped_inputs = image_data_tensor.view(-1, 1, 28, 28)
        output = classifier_network(reshaped_inputs.to(device))

        loss = loss_function(output, target.to(device))
        loss_tmp += loss.mean().item()

        counter += 1
        if counter % 500 == 0:
            progress.append(loss_tmp / 500)
            loss_tmp = 0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'epoch = {i + 1}, counter = {counter}')
