import numpy
from numpy import array
from tabulate import tabulate
from numpy import linalg


A = array([[5.554,0.252,0.496,0.237],[0.580,4.953,0.467,0.028],
           [0.319,0.372,8.935,0.520],[0.043,0.459,0.319,4.778]])
B = array([[0.442],[0.464],[0.979],[0.126]])


def symmetization():
    print('1. Провести симметризацию системы')
    Transposed = A.transpose()
    print('\nТранспонированная матрица A:')
    print(tabulate(Transposed, tablefmt='fancy_grid'))
    Multiplied = numpy.matmul(Transposed, A)
    print('\nПроизведение AT*A:')
    print(tabulate(Multiplied, tablefmt='fancy_grid'))
    print('\nПроизведение AT и матрицы свободных членов:')
    Multiplied = numpy.matmul(Transposed, B)
    print(tabulate(Multiplied, tablefmt='fancy_grid'))


def main():
    symmetization()


main()
