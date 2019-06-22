import numpy
from numpy import array
from tabulate import tabulate
from numpy import linalg
import math
import sys
from functools import reduce

A = array([[3.452, 0.458, 0.125, 0.236], [0.254, 2.458, 0.325, 0.126], [0.305, 0.125, 3.869, 0.458], [0.423, 0.452, 0.248, 3.896]])
 
B = array([0.745, 0.789, 0.654, 0.405]).transpose()


def symmetization():
    Transposed = A.transpose()
    Multiplied = numpy.matmul(Transposed, A)
    return Multiplied


def isDiagonal(S):
    n = len(S)
    for i in range(0, n):
        for j in range(0, n):
            if S[i][i] == 0 or (S[i][j] != 0 and j != i):
                return False
    return True


def printDiagonal(S):
    n = len(S)
    print('\nСобственные значения: \n')
    for i in range(0, n):
        print('Собственный значение ', i + 1, ' = ', S[i][i])


def eig(S, e):
    n = len(S)
    U_list = []
    print('\nНайти собственные значения матрицы методом вращений')
    print('\nСимметрическая матрица: ')
    print(tabulate(S, tablefmt='fancy_grid'))
    iter = 1
    while (True):
        max = sys.float_info.min
        for i in range(0, n):
            for j in range(0, n):
                if math.fabs(S[i][j]) > max and i != j and i < j:
                    max = math.fabs(S[i][j])
                    max_i = i
                    max_j = j
        if e > max:
            break
        alpha = 0.5 * \
            math.atan(2 * S[max_i][max_j] /
                      (S[max_i][max_i] - S[max_j][max_j]))
        U = numpy.zeros((n, n))
        for i in range(0, n):
            U[i][i] = 1
        U[max_i][max_i] = math.cos(alpha)
        U[max_j][max_j] = math.cos(alpha)
        U[max_i][max_j] = -math.sin(alpha)
        U[max_j][max_i] = math.sin(alpha)
        print('\nИтерация', iter)
        print('\nИндекс i =', max_i + 1, 'Индекс j = ', max_j + 1)
        print('Угол =', alpha)
        print('cos(угла) =', math.cos(alpha), ', sin(угла) =', math.sin(alpha))
        print('\nМатрица U:')
        print(tabulate(U, tablefmt='fancy_grid'))
        U_list.append(U)
        S = numpy.matmul(numpy.matmul(U.transpose(), S), U)
        iter += 1
        print('\nМатрица A:')
        print(tabulate(S, tablefmt='fancy_grid'))
    printDiagonal(S)
    print('\nСобственные вектора: ')
    X = reduce((lambda x, y: numpy.matmul(x, y)), U_list)
    print(tabulate(X, tablefmt='fancy_grid'))
    for i in range(0, n):
        print('\n X -', i + 1, '\n')
        current = X[i][:, numpy.newaxis]
        print(tabulate(current, tablefmt='fancy_grid'))


def main():
    eig(symmetization(), 0.001)
    W, V = linalg.eig(symmetization())
    print('Решение с помощью linalg.eig: ')
    print(tabulate(V, tablefmt='fancy_grid'))
    print(W)


main()
