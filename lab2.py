from numpy import array
from numpy import linalg
from tabulate import tabulate
import math
import numpy

A = array([[3.0, 1.0, 1.0],[1.0 ,4.0 , 1.0], [1.0, 1.0, 5.0]])
# A = array([[3.738,0.195,0.275,0.136],[0.519,5.002,0.405,0.283],
#           [0.306,0.381,4.812,0.418],[0.272,0.142,0.314,3.935]])
B = array([[1.0, -1.0, 1.0]])
# B = array([[0.815],[0.191],[0.423],[0.352]])
n = len(A)
B = B.transpose()
print(B)

print(A)


def replace_b():
    S = numpy.copy(B)
    for i in range(0, len(B)):
        S[i] = S[i] / A[i][i]
    return S


def replace_a():
    S = numpy.copy(A)
    for i in range(0, n):
        div = - A[i][i]
        for j in range(0, n):
            if j != i:
                S[i][j] = S[i][j] / div
            else:
                S[i][j] = 0
    return S


def solve_simple(k):
    print("\n3. Решить систему методом простых итераций")
    new_A = replace_a()
    new_B = replace_b()
    X = numpy.zeros((n, 1))
    Res = numpy.copy(B)
    for step in range(0, k):
        for i in range(0, n):
            X[i] = numpy.sum(new_A[i].dot(Res)) + new_B[i]
        Res = numpy.copy(X)
        print("\nИтерация ", step + 1)
        print(tabulate(X, tablefmt='fancy_grid'))


def mpi():
    print("1. Преобразовать систему к виду, необходимому для применения метода простых итераций. "
          "Проверить условия сходимости МПИ")
    print("\nПреобразованная матрица B: ")
    new_B = replace_b()
    print(tabulate(new_B, tablefmt='fancy_grid'))
    print("\nПреобразованная матрица A: ")
    new_A = replace_a()
    print(tabulate(new_A, tablefmt='fancy_grid'))
    W = linalg.eigvals(new_A)
#   print(W)
    flag = 0
    for i in range(0, n):
        if abs(W[i]) < 1:
            flag += 1
    if flag == n and linalg.norm(new_A) < 1:
        print("\nРешения собственных значений матрицы по модулю < 1")
        print("\nНорма матрицы", linalg.norm(new_A), " < 1")
        print("\nСледоательно, необходимое и достаточное условия МПИ выполняеются")
    else:
        print("\nУсловия не выполняются")


def steps():
    print("\n2. Найти необходимое число итеративных шагов для решения системы методом простой итерации с точностью 0,01")
    new_A = replace_a()
    print(abs(linalg.norm(new_A.dot(B))))
    k = math.ceil(math.log(0.01 * (1 - linalg.norm(new_A)) / abs(linalg.norm(new_A.dot(B))), linalg.norm(new_A)))
    print("\nКоличество шагов: ", k)
    return k


def zeildel_check():
    print("\n4.Преобразовать систему к виду, необходимому для применения метода Зейделя. Проверить условия сходимости "
          "метода Зейделя")
    flag = 0
    print("\nПреобразованная матрица:")
    new_A = replace_a()
    print(tabulate(new_A, tablefmt='fancy_grid'))
    for i in range(0, n):
        if max(A[i]) == A[i][i]:
            flag += 1
    if flag == n and linalg.norm(new_A) < 1:
        print("\nМетод Зейделя сходится, диагональные элементы доминируют по сторкам и столбцам и норма матрицы < 1")
    else:
        print("\nНевозможно применить метод Зейделя")


def zeildel_solve(k):
    print("\n5. Решить систему методом Зейделя")
    new_A = replace_a()
    new_B = replace_b()
    X = numpy.zeros((n, 1))
    Res = numpy.copy(B)
    for step in range(0, k):
        for i in range(0, n):
            X[i] = numpy.sum(new_A[i].dot(Res)) + new_B[i]
            Res[i] = X[i]
        print("\nИтерация ", step + 1)
        print(tabulate(Res, tablefmt='fancy_grid'))


def main():
    mpi()
    k = steps()
    solve_simple(k)
    zeildel_solve(k)
    print("\nРешение linalg.solve:")
    print(tabulate(linalg.solve(A, B), tablefmt='fancy_grid'))


main()







