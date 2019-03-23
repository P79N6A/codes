# coding:utf-8

import torch
import numpy as np
# print(torch.Tensor(100))
a = np.array([1])
b = torch.from_numpy(a)
b = b.squeeze()
print(b.item())
print(b.data)
print(type(b.data))
