import torchvision

# 通过torch 官方渠道下载数据集
train_dataset = torchvision.datasets.MNIST(
    root='../data/mnist',
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='../data/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
