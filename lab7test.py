from matplotlib import pyplot as plt
from math import pi
import sympy as sp
import numpy as np
from scipy import integrate


x = sp.Symbol('x')
expr = x ** 2 * sp.log(x)
# expr = 1 / 12 * x ** 4 + 1 / 3 * x - 1 / 60
my_func = sp.utilities.lambdify(x, expr, 'numpy')

a = 1
b = 2

def trapezoidal(a, b, func, n):
    h = 1.0 * (b - a) / n
    summ = (func(a) + func(b)) * 0.5
    x0 = 0
    for i in range(1, n + 1):
        x1 = a + i * h
        # plt.fill_between([x0, x1],[func(x0), func(x1)])
        summ += func(x1)
        x0 = x1
    # plt.show()

    return summ * h
    

def auto_step_select(a, b, eps, method):
    S = 0
    n = 1
    while True:
        S0 = S
        n = 2 * n    
        S = method(a, b, my_func, n)
        if (abs(S - S0)) <= eps:
            break

    return S, n


def simpson(a, b, func, n):
    if n % 2 == 1:
        n += 1
    dx = 1.0 * (b - a) / n
    summ = (func(a) + 4 * func(a + dx) + func(b))
    for i in range(1, int(n / 2)):
        summ += 2 * func(a + (2 * i) * dx) + 4 * func(a + (2 * i + 1) * dx)

    return summ * dx / 3


def newton_leibnic():
    return integrate.quad(my_func, a, b)[0]


def find_max(power=2):
    diff = sp.diff(expr, x, power)
    new_func = sp.utilities.lambdify(x, diff, modules=['numpy'])
    x_points = sp.solve(diff, x)
    fit = [cur for cur in x_points if a < cur < b]
    points = list(map(lambda point: new_func(float(point)), fit))
    if not points:
        return max(new_func(a), new_func(b))
    return max(new_func(a), new_func(b), max(points))


def print_method_info(title, step='h'):
    if 'Симпсона' in title:
        result, n = auto_step_select(a, b, 0.001, simpson)
        h = 1.0 * (b - a) / n
        maximum = find_max(4)
        if step == 'h^2':
            h = h * 2
            n = n * 2
            result = simpson(a, b, my_func, n)
        ep = 1.0 * (b - a) * h ** 4 / 2880 * maximum
    if 'Трапеций' in title:
        result, n = auto_step_select(a, b, 0.001, trapezoidal)
        maximum = find_max()
        h = 1.0 * (b - a) / n
        if step == 'h^2':
            h = h * 2
            n = n * 2
            result = trapezoidal(a, b, my_func, n)
        ep = 1.0 * (b - a) ** 3 / (12 * n ** 2) * maximum
    print('Метод ' + title + ':', result)
    print('Длина шага:', h)
    print('Оценка погрешности:', ep)
    print('Погрешность + результат', result - ep)
    print()
    

if __name__ == "__main__":
    print_method_info('Трапеций')
    print_method_info('Симпсона')
    print_method_info('Трапеций с шагом 2h', 'h^2')
    print_method_info('Симпсона с шагом 2h', 'h^2')
    print('Ньютон-Лейбниц:', newton_leibnic())
    