import numpy as np


def calc_weight(i, weights):
    return np.sum(weights[i.astype(bool)])


def calc_value(i, values):
    return np.sum(values[i.astype(bool)])


def print_geno(i):
    w = calc_weight(i, weights)
    v = calc_value(i, values)
    invalid = w > capacity
    if invalid:
        v = 0
    print(f"Weight: {w:>3d}, Value: {v:>3d}, {'invalid' if invalid else 'valid'}")


capacity = 40
weights = np.array([12, 10, 9, 16, 12, 2, 5, 3, 4])
values = np.array([4, 3, 5, 5, 6, 1, 3, 2, 1])
x = np.array([1, 1, 1, 0, 0, 1, 1, 0, 0])
y = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1])
z1 = np.array([1, 1, 1, 0, 1, 1, 0, 0, 1])
z2 = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0])
print_geno(x)
print_geno(y)
print_geno(z1)
print_geno(z2)
