import numpy
from numpy import array
from tabulate import tabulate
from numpy import linalg
import sys
import math


A = array([[3.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 5.0]])
#A = array([[3.738,0.195,0.275,0.136],[0.519,5.002,0.405,0.283],[0.306,0.381,4.812,0.418],[0.272,0.142,0.314,3.935]])


B = array([[1.00], [-1.0], [1.0]]) 
#B = array([[0.815], [0.191], [0.423], [0.352]])
n = len(A)


def symmetrize():
    return numpy.matmul(A.transpose(), A)


def isPositive(S):
    if linalg.eigvals(S).all() > 0:
        return True
    return False


def symmetization():
    print('1. Провести симметризацию системы')
    print(tabulate(A, tablefmt='fancy_grid'))
    Transposed = A.transpose()
    print('\nТранспонированная матрица A:')
    print(tabulate(Transposed, tablefmt='fancy_grid'))
    Multiplied = symmetrize()
    print('\nПроизведение AT*A, симметрическая матрица A:')
    print(tabulate(Multiplied, tablefmt='fancy_grid'))
    print('\nПроизведение AT и матрицы свободных членов, симметрическая матрица свободных членов:')
    Multiplied2 = numpy.matmul(Transposed, B)
    print(tabulate(Multiplied2, tablefmt='fancy_grid')) 
    return Multiplied


def create_matrix(V):
    S = numpy.zeros((n, n))
    S[0][0] = math.sqrt(V[0][0])
    for j in range(1, n):
        S[0][j] = V[0][j] / S[0][0] 
    for i in range(1, n):
        for j in range(i, n):
            item = 0.0
            for k in range(0, i):
                item += (S[k][i] * S[k][j])
            if i != j:
                S[i][j] = (V[i][j] - item) / S[i][i]
            else:       
                S[i][i] = math.sqrt(V[i][i] - item)
    return S

        
def square(V):
    print('\n2. Решение системы методом квадратного корня')
    S = create_matrix(V)
    print('\nНовая верхняя треугольная матрица')
    print(tabulate(S, tablefmt='fancy_grid'))
    test = S.transpose()
    print('\nНовая транспонированная треугольная матрица')
    print(tabulate(test, tablefmt='fancy_grid'))
    Y = numpy.matmul(linalg.inv(test), B)
    print('\nРешение матрицы Y')
    print(tabulate(Y, tablefmt='fancy_grid'))    
    X = numpy.matmul(linalg.inv(S), Y)
    print('\nРешение матрицы X')
    print(tabulate(X, tablefmt='fancy_grid'))
    print(tabulate(numpy.matmul(test, S),tablefmt='fancy_grid'))



def determinator(V):
    print('\n3. Вычислить определитель матрицы A')
    S = create_matrix(V)
    det = 1
    for i in range(0, n):
        det *= S[i][i] * S[i][i]
    print('\nОпределитель матрицы = ', det)


def invertion(V):
    print('\n4. Найти обратную матрицу A с помощью квадратного корня\n')
    S = create_matrix(V)
    Transposed = S.transpose()
    Inverted = numpy.zeros((n, n))
    for i in range(0, n):
        zero_matrix = numpy.zeros((n, 1))
        zero_matrix[i] = 1
        Y =  numpy.matmul(linalg.inv(Transposed), zero_matrix)
        X =  numpy.matmul(linalg.inv(S), Y)
        print('Итерация ', i + 1, ' Y, X')
        print(tabulate(Y, tablefmt='fancy_grid'))
        print(tabulate(X, tablefmt='fancy_grid'))
        for k in range(0, n):
            Inverted[k][i] = X[k]
    print('\nОбратная матрица\n')
    print(tabulate(Inverted, tablefmt='fancy_grid'))


if __name__ == "__main__":
    if isPositive(A) == False and linalg.det(A) != 0:
        print("Матрица не является положительно определенной")
    else:
        square(A)
        determinator(A)
        invertion(A)
        print('Инверсия linalg.inv')
        print(tabulate(linalg.inv(A), tablefmt='fancy_grid'))