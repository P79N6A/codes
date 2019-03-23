# coding:utf-8
import numpy as np
a = np.array([1, 2, 3])
b = a * 2
c = np.concatenate((a, b), axis=0)
print(a, b, c)
