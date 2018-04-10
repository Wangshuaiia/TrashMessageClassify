# coding=utf-8
from time import time
d = 0.0
d1 = time()

for i in range(200000000):
    pass
d2 = time()
d = (d2 - d1)
print(d)
