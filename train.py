import torch.optim
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *
import time

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
resize = 224  # AlexNet输入为224*224的图像，因此需要将输入图像变换大小
batch_size = 32  # 每一次小批量数目
num_workers = 0  # 是否开启多线程 Windows下请设置为0
lr = 0.01  # 学习率
num_epochs = 10  # 迭代次数
num_iter = 0  # 总训练样本数
test_acc = 0  # 测试集总正确数

# 导入数据
transform = torchvision.transforms.Compose(
    [torchvision.transforms.Resize(resize), torchvision.transforms.ToTensor()])  # 将图像变换为224*224

train_data = torchvision.datasets.FashionMNIST(root='../data', download=True, transform=transform)
test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(dataset=test_data, shuffle=True, batch_size=batch_size, num_workers=num_workers)

train_len = len(train_data)  # 训练集总大小
test_len = len(test_data)  # 测试集总大小

# 调用网络和损失函数
net = AlexNet()  # 定义网络为model中的AlexNet
net = net.to(device)  # 将网络转移到device上

loss_fn = nn.CrossEntropyLoss()  # 定义损失函数为交叉熵损失
loss_fn = loss_fn.to(device)  # 将损失函数转移到device上
# 优化器
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练开始
# writer = SummaryWriter("../summarywriter")  # 将训练过程保存到tensorboard展示
start_time = time.time()  # 计时开始
print("training on:{}".format(device))  # 显示训练设备
for i in range(num_epochs):
    print("----------第{}轮训练开始--------".format(i + 1))
    test_acc = 0  # 将测试集准确率置0
    # train
    net.train()  # 开启训练模式
    for data in train_loader:  # 从训练集中取出一个batch的数据 大小为batch_size*c*h*w
        images, labels = data  # 将数据分为图片和标签
        images = images.to(device)  # 将数据转移到device上
        labels = labels.to(device)
        outputs = net(images)  # 将图像放入网络得到输出
        loss = loss_fn(outputs, labels)  # 计算损失
        optimizer.zero_grad()  # 优化器将上一轮的梯度置0
        loss.backward()  # 反向传播
        optimizer.step()  # 优化器迭代，改变参数
        num_iter = num_iter + 1  # 总训练次数+1
        if num_iter % 100 == 0:  # 每训练100次，输出当前训练次数和损失
            print("num_iter:{} loss:{}".format(num_iter, loss))
            # writer.add_scalar('train loss', loss.item(), num_iter)
    # test
    net.eval()  # 开启评估模式
    with torch.no_grad():  # 在无梯度计算的环境下进行评估
        for data in test_loader:  # 从测试集中取出一个batch的数据 大小为batch_size*c*h*w
            images, labels = data  # 将数据分为图片和标签
            images = images.to(device)  # 将数据转移到device上
            labels = labels.to(device)
            outputs = net(images)  # 将图像放入网络得到输出
            accuracy = (outputs.argmax(1) == labels).sum()  # 计算概率最大的类别是否正确，并将结果相加
            test_acc = test_acc + accuracy  # 测试集正确数加上一个batch数据的正确数
    print("test accuracy:{}".format(test_acc / test_len))  # 计算测试集总正确率并输出
    # writer.add_scalar('test acc', test_acc, num_iter)

end_time = time.time()  # 结束计时
print("Time cost:{}s".format(end_time - start_time))  # 输出训练时间
