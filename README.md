# AlexNet-Pytorch
A pytorch implementations for AlexNet

The introduction to AlexNet can be found in [link](https://zhuanlan.zhihu.com/p/376526296)
# 1.Settings
Using these steps to prepare.

All dependencies can be installed into a conda environment with the provided environment.yaml file.
## 1.1Clone the repository
```git clone https://github.com/EightBarren7/AlexNet-Pytorch.git```

```cd AlexNet-Pytorch```
## 1.2Create conda environment and activate
```conda env create -f environment.yaml```

```conda activate AlexNet```
# 2.Usage
## 2.1 Adjust the hyper-parameters
You can change the batch_size according to your GPUs.

Howrver,you should **set num_workers to 0 if you are using Windows**
## 2.2Training
If all preparations have been completed,run ```train.py```

```python train.py```
## 2.3Evaling
Run ```test.py```to eval on FashionMNIST dataset.

```python test.py```

Classifying raw images has not been provided yet, it may be updated after a while.
## 2.4Using Tensorboard
Firstly,you should remove the comments in the following lines

```# from torch.utils.tensorboard import SummaryWriter```

```# writer = SummaryWriter("../summarywriter")  # 将训练过程保存到tensorboard展示```

```# writer.add_scalar('train loss', loss.item(), num_iter)```

```# writer.add_scalar('test acc', test_acc, num_iter)```

```# writer.close()```
Then run ```train.py```

Type in terminal ```tensorboard --logdir='../summarywriter' --port=6006```and open the browser

The results can be found in web address:```http://localhost:6006```
## References
1.Zhongxin.,"AlexNet网络模型的PyTorch实现",zhihu.com,2021.
