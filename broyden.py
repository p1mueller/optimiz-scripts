import numpy as np
import sympy as sp

import newton
import successive_halving as sh

x, y = sp.symbols("x y")


def main():
    p0 = (3.3, 0.2)
    f = None
    # f = (0.5 * x - y) ** 4 + (y - 1) ** 2
    df0 = [1, 0]
    Hinv = [[5, 0], [0, 2]]
    df1 = [0, 1]
    p1, p2 = broyden(f, p0, df0, Hinv, df1)
    print(f"Point 1 = {sh.format_point(np.squeeze(p1))}")
    print(f"Point 2 = {sh.format_point(np.squeeze(p2))}")


def broyden(f, p0, df0, Hinv, df1):
    p0 = np.array(p0)[:, None]
    if f is None:
        df0, Hinv0, df1 = prepare(df0, Hinv, df1)
        p1 = p0 - Hinv0 @ df0
    else:
        fn, grad, Hinv = newton.prepare(x, y, f)
        df0 = grad(*np.squeeze(p0))
        Hinv0 = Hinv(*np.squeeze(p0))
        p1 = p0 - Hinv0 @ df0
        df1 = grad(*np.squeeze(p1))
    d = df1 - df0
    s = p1 - p0
    Hinv1 = Hinv0 - ((Hinv0 @ d - s) @ s.T @ Hinv0) / np.linalg.inv(s.T @ Hinv0 @ d)
    p2 = p1 - Hinv1 @ df1
    return p1, p2


def prepare(df0, Hinv, df1):
    return np.array(df0)[:, None], np.array(Hinv), np.array(df1)[:, None]


if __name__ == "__main__":
    main()
