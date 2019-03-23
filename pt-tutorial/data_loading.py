# coding:utf-8
'''
torch.save
torch.load
torch.nn.Module.load_state_dict
'''
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# Define model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for k, v in model.state_dict().items():
    print(k, v.shape)
# print("model's state dict:%s" % model.state_dict())
'''
conv1.weight torch.Size([6, 3, 5, 5])
conv1.bias torch.Size([6])
conv2.weight torch.Size([16, 6, 5, 5])
conv2.bias torch.Size([16])
fc1.weight torch.Size([120, 400])
fc1.bias torch.Size([120])
fc2.weight torch.Size([84, 120])
fc2.bias torch.Size([84])
fc3.weight torch.Size([10, 84])
fc3.bias torch.Size([10])
'''
for k, v in optimizer.state_dict().items():
    print(k, v)
'''
state {}
param_groups [{'lr': 0.001, 'momentum': 0.9, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'params': [4316057104, 4431995600, 4431995672, 4431995744, 4431995816, 4431995888, 4431995960, 4431996032, 4431996104, 4431996176]}]
'''


'''可以用第3种方法存储多个模型，etc gan, seq2seq'''
'''method 1'''
'''save'''
torch.save(model.state_dict(), './model/model.pt')
'''load'''
model = Net()
model.load_state_dict(torch.load('./model/model.pt'))
model.eval()
input = Variable(torch.randn(1, 3, 32, 32))
out = model(input)
print(out)


'''method 2'''
model = Net()
torch.save(model, './model/model2.pt')
model = torch.load('./model/model2.pt')
model.eval()  # must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
input = Variable(torch.randn(1, 3, 32, 32))
out = model(input)
print(out)


'''method 3'''
torch.save({
    "epoch": 'epoch',
    "model_state_dict": model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 'loss'
}, './model/model3.tar')
model = Net()
optimizer = optim.SGD()
checkpoint = torch.load('./model/model3.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()  # or model.train()
'''
Other items that you may want to save are the epoch you left off on, the latest recorded training loss, external torch.nn.Embedding layers, etc.
'''


'''Save on GPU, Load on CPU'''
torch.save(model.state_dict(), 'PATH')
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load('PATH', map_location=device))


'''Save on GPU, Load on GPU'''
torch.save(model.state_dict(), 'PATH')
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load('PATH'))
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model

'''Save on CPU, Load on GPU'''
torch.save(model.state_dict(), 'PATH')
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load('PATH', map_location="cuda:0"))  # Choose whatever GPU device number you want
model.to(device)
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
