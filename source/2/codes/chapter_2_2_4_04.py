from torch.utils.data import DataLoader

from chapter_2_2_4_03 import train_dataset

train_loader = DataLoader(train_dataset, batch_size=16)
print(next(iter(train_loader)))
