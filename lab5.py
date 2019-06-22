from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.misc import derivative
import sympy as sympy


# exp = 0.1 * x ** 2 - x * log(x)
# eq = Eq(exp)
# syst = [Eq(tan(x * y + 0.3), 1.5*x - 1), Eq(0.9 * x ** 2 + 2 * y ** 2, 2)]


def f(x):
    if x <= 0:
        return np.nan
    return 0.1 * x ** 2 - x * np.log(x)


def equations(p):
    x, y = p
    return (np.tan(x * y + 0.1) - 2 * x ** 2, 0.6 * x ** 2 + 2 * y ** 2 - 1)


def chord(a, b, eps):

    while (abs(b - a) >= eps):

        t = a + (f(b) * (b - a)) / (f(b) - f(a))
        if f(a) * f(t) < 0:
            b = t
        elif f(t) * f(b) < 0:
            a = t
        else:
            return t

    return t


def dx(num):
    return abs(0 - f(num))


def df(num):
    return derivative(f, num)


def newtons_method(x0, e):
    delta = dx(x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(x0)
    return x0


def equatation1(x):
    equatation = (-0.6 * x ** 2 + 1) / 2
    if equatation < 0:
        return np.nan
    return np.sqrt(equatation)


def equatation2(x):
    equatation = np.arctan(2 * x ** 2) - 0.1
    if equatation == 0:
        return np.nan
    return equatation / x


def graphic():
    x = linspace(-5, 5, 500)
    y = list(map(lambda num: f(num), x))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    ax1.axhline(y=0, color='k')
    ax1.axvline(x=0, color='k')
    ax1.grid(True, which='both')
    ax1.plot(1.1181808850489714, 0, 'ro')
    ax1.plot(x, y)
    x = linspace(-2, 2, 500)
    y1 = list(map(lambda num: equatation1(num), x))
    y2 = list(map(lambda num: equatation2(num), x))
    ax2.plot(x, y1)
    ax2.plot(x, y2)
    ax2.plot(0.463589211496136, 0.659965628027769, 'ro')
    ax2.plot(-0.108526, 0.704603, 'ro')
    ax2.grid(True, which='both')
    ax2.axhline(y=0, color='k')
    ax2.axvline(x=0, color='k')
    plt.show()


def jacobi(x_v):
    n = len(x_v)
    x, y = sympy.symbols('x y')
    y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
    y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
    eqs = [y1, y2]
    x_v_s = [x, y]
    J = np.identity(n)

    for i in range(n):
        for j in range(n):
            J[i][j] = sympy.diff(eqs[i], x_v_s[j]).subs(x_v_s[0], x_v[0]).subs(x_v_s[1], x_v[1])
    return J


def system_newton(init, eps, mod=False):
    x, y = init
    steps = 0
    if not mod:
        while True:
            steps += 1
            y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
            y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
            J = jacobi([x, y])
            solution = np.array([x, y]) - np.dot(np.linalg.inv(J), np.array([y1, y2]))
            if abs(solution[0] - x) < eps:
                break
            x, y = solution[0], solution[1]
    else:
        J = jacobi([x, y])
        while True:
            steps += 1
            y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
            y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
            solution = np.array([x, y]) - np.dot(np.linalg.inv(J), np.array([y1, y2]))
            if abs(solution[0] - x) < eps:
                break
            x, y = solution[0], solution[1]
    return solution, steps


def system_simple_iterations(init, end, eps):
    n = len(init)
    y1 = lambda x, y: sympy.tan(x * y + 0.1) - 2 * x ** 2
    y2 = lambda x, y: 0.6 * x ** 2 + 2 * y ** 2 - 1
    f_v = [y1, y2]
    solution = [0, 0]
    a = init
    end_x = end
    steps = 0
    while True:
        steps += 1
        for i in range(n):
            solution[i] = end_x[i] - f_v[i](end_x[0], end_x[1]) * (a[i] - end_x[i]) / (f_v[i](a[0], a[1]) - f_v[i](end_x[0], end_x[1]))
        if abs(solution[0] - end_x[0]) < eps:
            break
        end_x = solution.copy()
    return solution, steps


if __name__ == "__main__":

    print('\nNewtons method, Root is at: ', newtons_method(2.5, 0.001))

    print("\nChords method, Root is at:", chord(1, 5, 0.001))

    print("\nSolve by fsolve, scipy library", fsolve(f, 5))

    newton = system_newton((-0.1, 0.6), 0.001)
    print('\nNewton method:\nx y = {}\nsteps = {}'.format(newton[0], newton[1]))

    newton_mod = system_newton((-0.1, 0.6), 0.001, True)
    print('\nModified Newton method:\nx y = {}\nsteps = {}'.format(newton_mod[0], newton_mod[1]))

    simple = system_simple_iterations((-0.1, 0.6), (0.1, 0.2), 0.001)
    print('\nSimple Iterations method:\nx y = {}\nsteps = {}'.format(simple[0], simple[1]))

    x, y = fsolve(equations, (5, 5))
    print('\nSolve by fsolve, scipy library:', list((x, y)))

    graphic()