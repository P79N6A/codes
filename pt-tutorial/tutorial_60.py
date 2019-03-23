# coding:utf-8
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch


def p(s):
    print (s)


'''
阅读材料：
100+ Tensor的操作，包括换位、索引、切片、数学运算、线性算法和随机数等等。
详见：torch - PyTorch 0.1.9 documentation
'''
x = torch.Tensor(2, 3)
p(x)
x = torch.rand(2, 3)
p(x)
p(x.size())
p(x.shape)
'''
> tensor([[1.3733e-14, 6.4069e+02, 4.3066e+21],
        [1.1824e+22, 4.3066e+21, 6.3828e+28]])
>>tensor([[0.1079, 0.7251, 0.1244],
        [0.8968, 0.4241, 0.7044]])
>>torch.Size([2, 3])
>>torch.Size([2, 3])
'''

y = torch.rand(2, 3)
z = torch.add(x, y)
p(z)
z = torch.Tensor(2, 3)
torch.add(x, y, out=z)  # 直接存储在out
p(x)
p(y)
y.add_(x)
p(y)


p(y[:, 1])


'''
Tensor == (numpy)array转换
Torch的Tensor和numpy的array会共享他们的存储空间，修改一个会导致另外的一个也被修改。
'''
a = torch.ones(2, 3)
p(a)
p(a.numpy())

b = np.array([1, 2, 3])
p(torch.from_numpy(b))
np.add(b, 1, out=b)
p(b)

if torch.cuda.is_available():
    p('cuda is ok')
    a = a.cuda()


'''
autograd
grad can be implicitly created only for scalar outputs
阅读材料：
你可以在这读更多关于Variable 和 Function的文档: pytorch.org/docs/autograd.html
'''
x = Variable(torch.ones(2, 2), requires_grad=True)
y = (x + 2).mean()
# y.creator
p(y.grad_fn)
y.backward()
# d(y)/dx
p(x.grad)


'''model'''


class model1(nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        p(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.softmax(self.fc3(x), dim=0)
        return x


model = model1()
target = torch.randn(10, 10)
input = Variable(torch.randn(10, 1, 32, 32))
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
for i in range(10):
    p(i)
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    p('loss')
    p(loss)
    loss.backward()
    optimizer.step()


'''
图像：可以用Pillow, OpenCV，torchvision这个包可用,其中包含了一些现成的数据集如：Imagenet, CIFAR10, MNIST等等
CIFAR10：'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
声音：可以用scipy和librosa。
文本：处理使用原生Python或者Cython以及NLTK和SpaCy。
'''


'''test grad'''
model.zero_grad()
input = Variable(torch.randn(10, 1, 32, 32))
output = model(input)


p('conv1 grad before:')
p(model.conv1.bias.grad)
output.backward(torch.randn(10, 10))  # 使用随机的梯度进行反向传播
p('conv1 grad after:')
p(model.conv1.bias.grad)


target = torch.randn(10, 10)
criterion = nn.MSELoss()
loss = criterion(output, target)
p(loss.grad_fn)
p(loss.grad_fn.next_functions[0][0])  # 替换函数
model.zero_grad()


p('learning_rate')
learning_rate = 0.01
for f in model.parameters():
    # p(f.grad.data)
    p(f.grad.data.size())
    f.data.sub_(f.grad.data * learning_rate)


'''
model1(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=16*5*5, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''

'''test model'''
model = model1()
p(model)
params = list(model.parameters())
p(len(params))  # 长度为模型的层数
p(params[9].size())  # torch.Size([6, 1, 5, 5]) 第一个卷积的核参数
input = Variable(torch.randn(1, 1, 32, 32))
out = model(input)
p(out)


'''
view
'''
x = torch.Tensor(3, 4, 5)
y = x.view(x.size(0), -1)
p(y.shape)
z = x.view(-1, 20)
p(z.shape)
