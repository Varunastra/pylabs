import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math
import timeit
from scipy import integrate
from tabulate import tabulate

x, y = sp.symbols("x y")


def error_estimation(method, func, a, b, eps, p):
    n = 1
    while True:
        if abs(method(func, a, b, 2 * n) - method(func, a, b, n)) / (2 ** p - 1) < eps:
            break
        n *= 2
    return n * 2


def trapez(func, a, b, n):
    h = (b - a) / n
    return h * sum([(func(a + h * i) + func(a + h * (i + 1))) / 2 for i in range(n)])


def simpson(func, a, b, n):
    h = (b - a) / n
    return (
        h
        / 3
        * sum(
            [
                func(a + h * i) + 4 * func(a + h * (i + 1)) + func(a + h * (i + 2))
                for i in range(0, n - 1, 2)
            ]
        )
    )


def newton_leibniz(func, a, b):
    return integrate.quad(func, a, b)[0]


def de_error_estimation(ode_method, func, a, b, x0, y0, eps):
    n = 2
    while True:
        k, y1 = ode_method(func, a, b, x0, y0, n)
        k, y2 = ode_method(func, a, b, x0, y0, n // 2)
        if abs(y2[-1] - y1[-1]) / (1 / 15) < eps:
            break
        n *= 2
    return n


def runge(func, a, b, x0, y0, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        F1 = h * func(x[i], y[i])
        F2 = h * func(x[i] + h / 2, y[i] + F1 / 2)
        F3 = h * func(x[i] + h / 2, y[i] + F2 / 2)
        F4 = h * func(x[i] + h, y[i] + F3)
        y[i + 1] = y[i] + 1 / 6 * (F1 + F4 + 2 * (F2 + F3))
        x[i + 1] = x[i] + h
    return x, y


def adams(f, a, b, x0, y0, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = x0
    y[0] = y0
    x[1] = x[0] + h
    y[1] = y[0] + h * f(x[0], y[0])
    for i in range(1, n):
        p = y[i] + h / 2 * (3 * f(x[i], y[i]) - f(x[i - 1], y[i - 1]))
        x[i + 1] = x[i] + h
        y[i + 1] = y[i] + h / 2 * (f(x[i], y[i]) + f(x[i + 1], p))
    return x, y


def eiler(func, a, b, x0, y0, n):
    h = (b - a) / n
    x = np.empty(n + 1, float)
    y = np.empty(n + 1, float)
    x[0] = x0
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * func(x[i], y[i])
        x[i + 1] = x[i] + h
    return x, y


def print_runge(diff_func, diff_a, diff_b, x0, y0, nd):
    runge_x, runge_y = runge(diff_func, diff_a, diff_b, x0, y0, nd)
    runge_x_2, runge_y_2 = runge(diff_func, diff_a, diff_b, x0, y0, nd // 2)
    plt.plot(runge_x, runge_y, label="Рунге")
    table1 = [["xᵢ", "yᵢ", "ỹᵢ", "∆ᵢ = |yᵢ - ỹᵢ|"]]
    for i in range(len(runge_x)):
        if i % 2 == 0:
            table1.append(
                [
                    runge_x[i],
                    runge_y[i],
                    runge_y_2[i // 2],
                    abs(runge_y_2[i // 2] - runge_y[i]),
                ]
            )
    print("Метод Рунге:")
    print(tabulate(table1, tablefmt="fancy_grid"))


def print_adams(diff_func, diff_a, diff_b, x0, y0, nd):
    adams_x, adams_y = adams(diff_func, diff_a, diff_b, x0, y0, nd)
    adams_x_2, adams_y_2 = adams(diff_func, diff_a, diff_b, x0, y0, nd // 2)
    plt.plot(adams_x, adams_y, label="Адамс")
    table2 = [["xᵢ", "yᵢ", "ỹᵢ", "∆ᵢ = |yᵢ - ỹᵢ|"]]
    for i in range(len(adams_x)):
        if i % 2 == 0:
            table2.append(
                [
                    adams_x[i],
                    adams_y[i],
                    adams_y_2[i // 2],
                    abs(adams_y_2[i // 2] - adams_y[i]),
                ]
            )
    print("Метод Адамса:")
    print(tabulate(table2, tablefmt="fancy_grid"))


def print_eiler(diff_func, diff_a, diff_b, x0, y0, nd):
    eiler_x, eiler_y = eiler(diff_func, diff_a, diff_b, x0, y0, nd)
    eiler_x_2, eiler_y_2 = eiler(diff_func, diff_a, diff_b, x0, y0, nd // 2)
    plt.plot(eiler_x, eiler_y, label="Эйлер")
    table3 = [["xᵢ", "yᵢ", "ỹᵢ", "∆ᵢ = |yᵢ - ỹᵢ|"]]
    for i in range(len(eiler_x)):
        if i % 2 == 0:
            table3.append(
                [
                    eiler_x[i],
                    eiler_y[i],
                    eiler_y_2[i // 2],
                    abs(eiler_y_2[i // 2] - eiler_y[i]),
                ]
            )
    print("Метод Эйлера:")
    print(tabulate(table3, tablefmt="fancy_grid"))


if __name__ == "__main__":
    plt.grid()

    A = sp.exp(-sp.sqrt(x))
    func = sp.lambdify(x, A)
    a = 1
    b = 4
    eps = 0.001

    n = error_estimation(trapez, func, a, b, eps, 2)
    print("Метод трапеций:")
    t = trapez(func, a, b, n)
    print("Длина шага:", (b - a) / n)
    print("При шаге h:  ", t)
    t2 = trapez(func, a, b, n // 2)
    print("При шаге h/2:", t2)
    print()
    n = error_estimation(simpson, func, a, b, eps, 4)
    print("Метод Симпcона:")
    s = simpson(func, a, b, n)
    print("Длина шага:", (b - a) / n)
    print("При шаге h:  ", s)
    s2 = simpson(func, a, b, n // 2)
    print("При шаге h/2:", s2)
    f = newton_leibniz(func, a, b)
    print('Трапеции', abs(f - t))
    print('Симпсон', abs(f - s))
    print()
    print("Ньютон-Лейбниц:", f)
    print()

    diff_A = y * (0.5 * x * y - 1)
    diff_func = sp.lambdify((x, y), diff_A)
    diff_a = 0
    diff_b = 2
    x0 = 0
    y0 = 2

    nd = de_error_estimation(runge, diff_func, diff_a, diff_b, x0, y0, 0.0001)

    print_runge(diff_func, diff_a, diff_b, x0, y0, nd)

    print_adams(diff_func, diff_a, diff_b, x0, y0, nd)

    print_eiler(diff_func, diff_a, diff_b, x0, y0, nd)

    f = sp.Function("f")
    C1 = sp.Symbol("C1")
    solve = sp.simplify(sp.dsolve(sp.diff(f(x), x) + f(x) - 0.5 * x * f(x) ** 2, f(x)))
    print("Решение:", solve)
    CC = sp.solve(solve.subs({x: x0, f(x): y0}), C1)[0]
    print("Задача Коши:", solve.rhs.subs(C1, CC))
    solution = sp.lambdify(x, solve.rhs.subs(C1, CC))
    xL = np.linspace(diff_a, diff_b)
    plt.plot(xL, solution(xL), "k--", label="Точное решение")

    legend = plt.legend(loc="upper right", shadow=True, fontsize="medium", frameon=True)
    plt.show()

