# coding:utf-8
from itertools import zip_longest
for i in zip_longest(*[[1, 2, 3], [1]], fillvalue='0'):
    print(i)

for i in zip_longest([1, 2, 3], [1], fillvalue='0'):
    print(i)
