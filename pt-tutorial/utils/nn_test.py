# coding:utf-8
import torch
import torch.nn as nn
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
input = torch.randn(5, 3, 10)  # seq_length, batch_size, text vec dim
h0 = torch.randn(2, 3, 20)  # num_layers*num_directions, batch_size, hidden_size
output, hn = rnn(input, h0)
print (output.shape)
print (hn.shape)
'''
torch.Size([5, 3, 20])
torch.Size([2, 3, 20])
'''

'''
知乎：LSTM神经网络输入输出究竟是怎样的？
https://www.zhihu.com/question/41949741
'''
