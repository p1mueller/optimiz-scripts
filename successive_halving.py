import numpy as np
import sympy as sp


def main():
    x, y = sp.symbols("x y")

    beta = 1.0
    start_point = (2, 3)
    f = 4 * x ** 2 - 4 * x * y + 2 * y ** 2

    fn, grad = prepare(x, y, f)
    print("Successive halving:")
    print_solution(halving_step(np.array(start_point), fn, grad, beta))

    print("\nSuccessive halving with parabola fitting:")
    print_solution(halving_step_parabola(np.array(start_point), fn, grad, beta))

    # beta = 0.125
    # P0 = 7.5
    # PB = 2.2
    # P2B = 21.7
    # bs = beta_star(beta, P0, PB, P2B)
    # print(f"\nBeta_star = {bs:.5f}")


def beta_star(beta, P0, PB, P2B):
    a = (P0 - 2 * PB + P2B) / (2 * beta ** 2)
    b = (-3 * P0 + 4 * PB - P2B) / (2 * beta)
    return -b / (2 * a)


def halving_step(point, f, grad, beta=1.0):
    p0 = np.array(point)
    b = beta
    fp = f(*point)
    fp2 = fp + 1
    df = grad(*point)

    while fp <= fp2:
        p1 = p0 - b * df
        fp2 = f(*p1)
        if fp2 >= fp:
            b /= 2
    return p1, fp2, b


def halving_step_parabola(p0, f, grad, beta=1.0):
    p0 = np.array(p0)
    p1, fp1, b = halving_step(p0, f, grad, beta)
    dfp = grad(*p0)
    bs = beta_star(b, f(*p0), f(*p1), f(*(p0 - 2 * b * dfp)))
    p2 = p0 - bs * dfp
    fp2 = f(*p2)
    if fp2 < fp1:
        return p2, fp2, b
    return p1, fp1, b


def prepare(x, y, f):
    fn = sp.lambdify((x, y), f, modules="numpy")
    dx = sp.lambdify((x, y), f.diff(x), modules="numpy")
    dy = sp.lambdify((x, y), f.diff(y), modules="numpy")
    grad = lambda x, y: np.array([dx(x, y), dy(x, y)])
    return fn, grad


def format_point(p):
    return "(" + ", ".join([f"{v:.3f}" for v in p]) + ")"


def print_solution(args):
    p, fp, beta = args
    print(f"Next point = {format_point(p)}")
    print(f"Value at new point = {fp:.3f}")
    print(f"Step size = {beta:.3f}")


if __name__ == "__main__":
    main()
