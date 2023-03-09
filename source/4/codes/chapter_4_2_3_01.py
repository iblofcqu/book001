import torch

workers = 0
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
epochs = 100
lr = 0.0002
beta = 0.5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
