from torch.utils.data import DataLoader
import torch
import torchvision

# 选择device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])  # 将图像变换为224*224
test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

# 导入保存的模型
net = torch.load('AlexNet.pth')

# 开始测试
step = 0
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=1, num_workers=4)  # 取单张图片和标签
with torch.no_grad():
    for data in test_loader:
        image, label = data
        image = image.to(device)
        outputs = net(image)
        outputs = outputs.argmax(1).sum()  # 得到预测结果的索引值
        step = step + 1
        print('第{}张图片的预测结果为'.format(step), test_data.classes[outputs], '真实结果为', test_data.classes[label])
