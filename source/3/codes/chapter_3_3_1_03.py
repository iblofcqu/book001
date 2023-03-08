from torchvision import transforms

data_transforms = {
    'train': transforms.Compose([
        # 调整图片尺寸—>[224, 224]
        transforms.Resize((224, 224)),
        # 随机图片灰度化
        transforms.RandomGrayscale(p=0.1),
        # 随机仿射变化
        transforms.RandomAffine(0, shear=5, scale=(0.8, 1.2)),
        # 图片属性变换
        transforms.ColorJitter(brightness=(
            0.5, 1.5), contrast=(0.8, 1.5), saturation=0),
        # 随机图片水平翻转
        transforms.RandomHorizontalFlip(),
        # 将图片格式转换为张量
        transforms.ToTensor(),
        # 图片归一化
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
    # 测试集图片预处理，只进行图片尺寸、格式和归一化处理
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]),
}
