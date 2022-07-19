import numpy as np


def F(x, i):
    maskA = x == 0
    maskB = x == 1
    if np.sum(maskA) != np.sum(maskB):
        return -1
    return 100 - np.abs(np.sum(i[x == 0]) - np.sum(i[x == 1]))


i = np.array([30, 20, 10, 14, 6, 4, 12, 8, 1, 2])
x = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
y = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
z = np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
for v in (x, y, z):
    print(F(v, i))
