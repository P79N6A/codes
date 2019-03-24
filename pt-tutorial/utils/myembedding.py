# coding:utf-8
import torch
from torch.nn import Embedding
embedding = Embedding(10, 3, padding_idx=0)
print(embedding)
input = torch.LongTensor([[1, 2, 3, 4], [2, 3, 3, 4]])
print(embedding(input))
print(embedding.weight)  # 随机初始化的向量
