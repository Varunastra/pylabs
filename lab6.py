import sympy
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from math import sqrt

x0 = np.array([0.351, 0.867, 1.315, 2.013, 2.859, 4.0])
y0 = np.array([0.605, 0.218, 0.205, 1.157, 5.029, 3.5])
# x0 = np.array([0.0, 1.0, 2.0, 3.0])
# y0 = np.array([-2.0, -5.0, 0.0, -4.0])


def lagranzh():
    n = len(x0)
    x = sympy.symbols("x")
    mnogochlen = 0
    for i in range(0, n):
        mnojitel = y0[i]
        for j in range(0, n):
            if i != j:
                mnojitel *= (x - x0[j]) / (x0[i] - x0[j])
        mnogochlen += mnojitel
    mnogochlen = sympy.simplify(mnogochlen)

    return mnogochlen


def show_lagranzh(mnogochlen, multi=False):
    x = sympy.symbols("x")
    func = sympy.utilities.lambdify(x, mnogochlen, modules=["numpy"])
    if not multi:
        print("\nМногочлен Лагранжа: ", mnogochlen.evalf(5))
        print("L(x1 + x2) = ", mnogochlen.subs(x, x0[1] + x0[2]))
        plt.scatter(x0, y0, color="green")
    t = np.linspace(x0[0], x0[len(x0) - 1], 500)
    plt.plot(t, func(t), "-y")


def eval_raznost(formula="konechnaya"):
    n = len(x0)
    func = y0.copy()
    table = {}
    table["xk"] = x0
    table["yk"] = y0
    for i in range(0, n - 1):
        vertical_fill = []
        for j in range(0, n - i):
            try:
                if formula == "konechnaya":
                    value = round(func[j + 1], 5) - round(func[j], 5)
                else:
                    value = (func[j + 1] - func[j]) / (x0[j + 1 + i] - x0[j])
                vertical_fill.append(value)
            except IndexError:
                pass
        func = vertical_fill.copy()
        table[str(i + 1) + "-ого порядка"] = func
    return table


def polynom_newtona(table):
    n = len(x0)
    x = sympy.symbols("x")
    polynom = y0[0]
    for i in range(0, n - 1):
        mnojitel = 1
        for j in range(0, i + 1):
            mnojitel *= x - x0[j]
        polynom += table[str(i + 1) + "-ого порядка"][0] * mnojitel
    polynom = sympy.simplify(polynom)

    return polynom


def show_polynom_newtona(polynom, multi=False):
    x = sympy.symbols("x")
    func = sympy.utilities.lambdify(x, polynom, modules=["numpy"])
    t = np.linspace(x0[0], x0[len(x0) - 1], 500)
    if not multi:
        print("\nПолином Ньютона: ", polynom.evalf(5), '\n')
        plt.scatter(x0, y0, color="green")
    plt.plot(t, func(t), "-r")


def kusochno_linear():
    n = len(x0)
    systems = []
    x, y = sympy.symbols("x, y")

    for i in range(1, n):
        system = sympy.Matrix(((x0[i - 1], 1, y0[i - 1]), (x0[i], 1, y0[i])))
        solved = sympy.solve_linear_system(system, x, y)
        systems.append(solved[x] * x + solved[y])

    return systems


def show_linear(systems, multi=False):
    n = len(x0)
    x = sympy.symbols("x")
    if not multi:
        print('Кусочно-линейная интерполяция: \n')
        plt.scatter(x0, y0, color="green")
    for i in range(0, n - 1):
        if not multi: 
            if i / 2 != 1:
                print("       {", systems[i])
            else:
                print("F(x) = {", systems[i])
        function = sympy.utilities.lambdify(x, systems[i], modules=["numpy"])
        t = np.linspace(x0[i], x0[i + 1], 500)
        plt.plot(t, function(t), "-b")

    print()


def kusochno_kvadrat():
    n = len(x0)
    if n % 2 != 0:
        m = int(n / 2) + 1
    else:
        m = int(n / 2)
    x, y, z = sympy.symbols("x, y, z")

    systems = []
    for k in range(1, m):
        full_matrix = []
        for i in range(0, n - 2):
            full_matrix.append([x0[2 * k - i] ** 2, x0[2 * k - i], 1, y0[2 * k - i]])
        full_matrix = sympy.Matrix(full_matrix)
        solution = sympy.solve_linear_system(full_matrix, x, y, z)
        systems.append(solution[x] * x ** 2 + solution[y] * x + solution[z])

    return systems


def show_kvadrat(systems, multi=False):
    n = len(x0)
    if n % 2 != 0:
        m = int(n / 2) + 1
    else:
        m = int(n / 2)
    x = sympy.symbols("x")
    if not multi:
        print('Кусочно-квадратичная интерполяция \n')
        plt.scatter(x0, y0, color="green")
    for i in range(1, m):
        if not multi:
            if i == 1:
                print("F(x) = {", systems[i - 1])
            else:
                print("       {", systems[i - 1])
        function = sympy.utilities.lambdify(x, systems[i - 1], modules=["numpy"])
        t = np.linspace(x0[2 * i - 2], x0[2 * i], 500)
        plt.plot(t, function(t), "-m")

    print()


def multi_plot():
    systems = kusochno_kvadrat()
    show_kvadrat(systems, True)
    systems = kusochno_linear()
    show_linear(systems, True)
    polynom = lagranzh()
    show_lagranzh(polynom, True)
    show_cubic(True)
    plt.scatter(x0, y0, color="green")
    leg = plt.legend(
        [
            "Полином Лагранжа / Ньютона",
            "Кусочно-линейная",
            "Кусочно-квадратичная",
            "Кубический сплайн",
        ],
        fontsize=10,
    )
    colors = ["yellow", "blue", "magenta", "black"]
    for i, j in enumerate(leg.legendHandles):
        j.set_color(colors[i])
    plt.show()


def cubic_spline(argument):
    x = np.asfarray(x0)
    y = np.asfarray(y0)

    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    Li[0] = sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    index = x.searchsorted(argument)
    np.clip(index, 1, size - 1, index)

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    f0 = (
        zi0 / (6 * hi1) * (xi1 - argument) ** 3
        + zi1 / (6 * hi1) * (argument - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (argument - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - argument)
    )
    return f0


def show_cubic(multi=False):
    if not multi:
        plt.scatter(x0, y0, color="green")
    t = np.linspace(x0[0], x0[len(x0) - 1], 500)
    plt.plot(t, cubic_spline(t), "-k")


if __name__ == "__main__":
    polynom = lagranzh()
    show_lagranzh(polynom)
    plt.show()

    table = eval_raznost()
    headers = table.keys()
    print('\nТаблица конечных разностей: \n')
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign="center"))

    table = eval_raznost('razdelennaya')
    print('\nТаблица разделенных разностей: \n')
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign="center"))

    polynom = polynom_newtona(table)
    show_polynom_newtona(polynom)
    plt.show()

    systems = kusochno_linear()
    show_linear(systems)
    plt.show()

    systems = kusochno_kvadrat()
    show_kvadrat(systems)
    plt.show()

    show_cubic()
    plt.show()

    multi_plot()