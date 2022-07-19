import numpy as np
import sympy as sp

import successive_halving as sh

x, y = sp.symbols("x y")


def main():
    start_point = (6.5, 1.5)
    # f = 1 / 2 * x ** 2 + (y / 2 - 1) ** 2 + 3 * x + y + 5
    f = (0.5 * x - y) ** 4 + (y - 1) ** 2

    gp0, p1, fp1 = newton(start_point, f)
    print(f"Gradient = {sh.format_point(np.squeeze(gp0))}")
    print(f"New point = {sh.format_point(np.squeeze(p1))}")
    print(f"Value at new point = {np.squeeze(fp1):.3f}")


def newton(start_point, f):
    p0 = np.array(start_point)
    fn, grad, Hinv = prepare(x, y, f)
    gp0 = grad(*p0)
    p1 = p0[:, None] - Hinv(*p0) @ gp0
    return gp0, p1, fn(*p1)


def prepare(x, y, f):
    fn = sp.lambdify((x, y), f, modules="numpy")

    dx = f.diff(x)
    dy = f.diff(y)
    dxdy = dx.diff(y)
    dx2 = dx.diff(x)
    dy2 = dy.diff(y)

    grad = sp.lambdify((x, y), sp.Matrix([dx, dy]), modules="numpy")

    H = sp.Matrix([[dx2, dxdy], [dxdy, dy2]])
    Hinv = sp.lambdify((x, y), H.inv(), modules="numpy")
    return fn, grad, Hinv


if __name__ == "__main__":
    main()
