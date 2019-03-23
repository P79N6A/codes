# coding: utf-8
# %matplotlib inline
import os
from sklearn.metrics import classification_report, accuracy_score
import logging
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
logging.getLogger().setLevel(logging.INFO)


def p(s):
    print(s)


'''
《下载数据集》
注：这一部分需要下载部分数据集 因此速度可能会有一些慢 同时你会看到这样的输出

Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting tar file
Done!
Files already downloaded and verified'''
# p(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
# p(transforms.ToTensor())
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.imshow(npimg)
    # plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

p(' '.join('%5s' % classes[labels[j]] for j in range(4)))
imgshow(torchvision.utils.make_grid(images))


'''定义一个神经网络，
这个图像case：
relu 比 sigmoid loss 降得快，
不用激活函数，loss降得也很慢
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.conv2(x)
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.softmax(self.fc3(x), dim=0)
        # x = self.fc3(x)
        return x

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = F.max_pool2d(x, kernel_size=2, stride=2)
    #     x = self.conv2(x)
    #     x = F.max_pool2d(x, kernel_size=2, stride=2)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     # x = F.softmax(self.fc3(x), dim=0)
    #     # x = self.fc3(x)
    #     return x


def load_model(model_path):
    net = Net()
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
        net.eval()
    else:
        '''定义代价函数和优化器'''
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        '''train model'''
        epoch_num = 1
        for epoch in range(epoch_num):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                '''为什么要放进变量里呢？'''
                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss
                if i % 2000 == 1999:
                    logging.info('training: epoch: %s, data_iter: %s, loss:%s', epoch + 1, i + 1, running_loss / 2000)
                    running_loss = 0.0
        torch.save(net.state_dict(), model_path)
    return net


model_path = './model/picture_class.pt'
net = load_model(model_path)


dataiter = iter(testloader)
images, labels = dataiter.next()
p(' '.join('turth: %5s' % classes[labels[j]] for j in range(4)))

outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)
p(' '.join('predict: %5s' % classes[predicted[j]] for j in range(4)))

imgshow(torchvision.utils.make_grid(images))

'''整体准确率'''
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

logging.info(correct)
logging.info(type(correct))
logging.info(total)
logging.info(type(total))
logging.info('acc of the net on %s test images :%s', total, 1.0 * correct / total)

'''每个类别的准确率,召回率'''
true_labels = np.array([])
predict_labels = np.array([])

for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    true_labels = np.concatenate((true_labels, labels.numpy()), axis=0)
    predict_labels = np.concatenate((predict_labels, predicted.numpy()), axis=0)

report = classification_report(true_labels, predict_labels, target_names=classes)
p(report)
acc = accuracy_score(true_labels, predict_labels)
p(acc)

'''
acc: 0.4429
recision    recall  f1-score   support

      plane       0.66      0.36      0.47      1000
        car       0.51      0.74      0.61      1000
       bird       0.35      0.25      0.29      1000
        cat       0.38      0.11      0.17      1000
       deer       0.44      0.31      0.36      1000
        dog       0.37      0.59      0.45      1000
       frog       0.52      0.63      0.57      1000
      horse       0.54      0.54      0.54      1000
       ship       0.62      0.53      0.57      1000
      truck       0.39      0.65      0.49      1000

avg / total       0.48      0.47      0.45     10000
'''
