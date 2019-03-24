# coding:utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
'''
ref: https://zhuanlan.zhihu.com/p/34418001
'''

batch_size = 2
max_length = 3
hidden_size = 2
n_layers = 1
# seq_lengths 一定要是从大到小排序
tensor_in = torch.FloatTensor([[1, 2, 3], [1, 2, 0], [1, 0, 0]]).resize_(3, 3, 1)
print(tensor_in)
print(tensor_in.shape)
tensor_in = Variable(tensor_in)  # [batch, seq, feature], [2, 3, 1]
seq_lengths = [3, 2, 1]  # list of integers holding information about the batch size at each sequence step

# pack it
pack = pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
print('pack')
print(pack)
print(pack.batch_sizes)
print(pack.data)
pad = pad_packed_sequence(pack, batch_first=True)
print('pad')
print(pad)

# # initialize
# rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True)
# h0 = Variable(torch.randn(n_layers, batch_size, hidden_size))
#
# # forward
# out, _ = rnn(pack, h0)
#
# # unpack
# unpacked = pad_packed_sequence(out)
# print('111', unpacked)
