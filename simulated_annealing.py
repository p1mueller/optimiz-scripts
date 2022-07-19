import numpy as np


def main():
    p0 = [5.8, 8.9, 0]
    p1 = [6.2, 8, 0]
    T = 20
    f = lambda x, y, z: (2 * x - y) ** 2 + np.exp(np.abs(z)) + x ** 2

    p0, p1, T = prepare(p0, p1, T)
    prob = accept_probability(p0, p1, f, T)
    print(prob)


def prepare(p0, p1, T):
    return np.array(p0, float), np.array(p1, float), float(T)


def accept_probability(p0, p1, f, T):
    return min(1.0, np.exp((f(*p0) - f(*p1)) / T))


if __name__ == "__main__":
    main()
