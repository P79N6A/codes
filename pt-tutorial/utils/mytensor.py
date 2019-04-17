# coding:utf-8

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
# print(torch.Tensor(100))
# a = np.array([1])
# b = torch.from_numpy(a)
# b = b.squeeze()
b = torch.randn(1, 4)
print(b.shape)
print(b)
b = b.view(4, 1)
# b = b.squeeze()
print(b)
print(b.shape)
# print(b.item())
# print(b.data)
# print(type(b.data))


print(torch.tensor([1, 2, 3]))
print(torch.LongTensor([1, 2, 3]))

print(torch.tensor([1, 2, 3]) * torch.tensor([1, 2, 3]))
'''
tensor([1, 4, 9])
'''
print(torch.tensor([1, 2, 3]) * torch.tensor([[1], [2], [3]]))
'''
tensor([[1, 2, 3],
        [2, 4, 6],
        [3, 6, 9]])
'''
print(torch.dot(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3])))
'''
dot只能处理1维的
tensor(14)
'''
# print(torch.dot(torch.tensor([[1, 2, 3], [2, 3, 4]]), torch.tensor([1, 2, 3])))
'''
RuntimeError: dot: Expected 1-D argument self, but got 2-D
'''

print(torch.sum(torch.tensor([1, 2, 3]) * torch.tensor([[1], [2], [3]]), dim=0))


'''
transpose
'''
print('transpose')
x = torch.randn(2, 3)
print(x)
'''
tensor([[ 0.0925,  0.5096, -0.4221],
        [-0.1131,  0.1144, -0.6749]])
'''
print(torch.transpose(x, 0, 1))
'''
tensor([[ 0.0925, -0.1131],
        [ 0.5096,  0.1144],
        [-0.4221, -0.6749]])
'''
print(x)
'''
tensor([[ 0.0925,  0.5096, -0.4221],
        [-0.1131,  0.1144, -0.6749]])
'''
print(torch.transpose(x, 1, 0))
'''
tensor([[ 0.0925, -0.1131],
        [ 0.5096,  0.1144],
        [-0.4221, -0.6749]])
'''
'''
torch 的transpose只能两维度转置
permute可以多维换
'''
img_nhwc = torch.randn(10, 480, 640, 3)
print(img_nhwc.size())
img_nchw = img_nhwc.permute(0, 3, 1, 2)
print(img_nchw.shape)

'''bmm torch.bmm(batch1, batch2, out=None) → Tensor  batch1,2 三维If batch1 is a (b \times n \times m)(b×n×m) tensor, batch2 is a (b \times m \times p)(b×m×p) tensor, out will be a (b \times n \times p)(b×n×p) tensor.'''
print('bmm')
a = torch.randn(10, 2, 3)
b = torch.randn(10, 3, 2)
c = torch.bmm(a, b)
for _ in [a, b, c]:
    print(_.shape)
    # print(_)
'''
torch.Size([10, 2, 3])
torch.Size([10, 3, 2])
torch.Size([10, 2, 2])
'''

'''torch.cat'''
x = torch.randn(1, 3)
print(x)
xx = torch.cat((x, x, x), 0)
'''
tensor([[ 0.3048, -0.7201, -0.8386],
        [ 0.3048, -0.7201, -0.8386],
        [ 0.3048, -0.7201, -0.8386]])
'''
xy = torch.cat((x, x, x), 1)
'''tensor([[ 0.3048, -0.7201, -0.8386,  0.3048, -0.7201, -0.8386,  0.3048, -0.7201,
         -0.8386]])'''
print(xx)
print(xy)
print(torch.cat((torch.randn(1, 3), torch.randn(1, 2)), 1))

'''softmax dim: reduce dim'''
print('softmax dim')
a = torch.Tensor([[1, 2, 3], [3, 4, 5]])
print(a)
print(F.softmax(a, dim=0))
print(a.type())

''''''

decoder_input = torch.LongTensor([[0 for _ in range(5)]])
print(decoder_input.shape)

a = torch.randn(10, 5, 500) * torch.randn(1, 5, 500)  # shape: 10,5,500
a = torch.sum(a, dim=2)  # shape 10,5,1 减少一个维度  10,5
print(a.shape)


''''''
a = F.softmax(torch.randn(5, 10), dim=1)
print(a.shape)


a = torch.randn(10, 5, 500).transpose(0, 1)
print(a.shape)


a = torch.Tensor([1, 2, 3, 4, 5])
a.sum()
print(a)
print(a.sum())


'''gather'''
inp = torch.randn(5, 10)
target = torch.tensor([1, 4, 2, 3, 5])
a = torch.gather(inp, 1, target.view(-1, 1))
print(inp)
print(a)
print(target)


'''torch.masked_select'''
print('-' * 50 + 'torch.masked_select')
x = torch.randn(2, 3)
mask = x.ge(0.5)
y = torch.masked_select(x, mask)
print(x)
print(mask)
print(y)


'''torch.nn.utils.clip_grad_norm_
 梯度过大时，乘以比例，减小梯度。
 梯度太小，弥散时？
 https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
'''
print('-' * 50 + 'torch.nn.utils.clip_grad_norm_')
a = Variable(torch.tensor([[0.00001, 1000000000000]]), requires_grad=True)
print(a)
y = (a + 2).mean()
# y.creator
print(y.grad_fn)
y.backward()
# d(y)/da
print(a.grad)
torch.nn.utils.clip_grad_norm_(a, max_norm=0.1)


''''''
# a = input('>')
# print(a)

''''''
print('-' * 50 + 'test')
print(torch.ones(1, 1,  dtype=torch.long) * 0)

decoder_output = torch.randn(5, 4)
print(decoder_output)
print(torch.max(decoder_output, dim=1))

print(torch.cat((torch.tensor([1]), torch.tensor([2])), dim=0))

'''
https://zhuanlan.zhihu.com/p/31495102
expand
'''
x = torch.Tensor([[1], [2], [3]])
print(x.size())
x = x.expand(3, 4)
print(x)


a = [1, 2, 3, 4, 5]
print(a[::-1])

'''
full
'''
print(torch.full((2, 3), 3))


x = torch.randn(1, 3)
print(torch.cat((x, x)))



