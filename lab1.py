import numpy
from numpy import append
from numpy import array
from tabulate import tabulate
from numpy.linalg import inv


A = array([[5.554,0.252,0.496,0.237],[0.580,4.953,0.467,0.028],
           [0.319,0.372,8.935,0.520],[0.043,0.459,0.319,4.778]])
B = array([[0.442],[0.464],[0.979],[0.126]])

if numpy.linalg.det(A) != 0:
    print("Определить матрицы != 0, решения есть")
else:
    print("Опеделить матрицы = 0, решений нет")
    exit(0)
print('Решение методом Гаусса:')


def gauss(A):
    n = len(A)

    for i in range(0, n):

        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                A[k][j] += c * A[i][j]
        if(i<3):
            print('\nИтерация номер {}\n'.format(i+1))
            print(tabulate(A, tablefmt='fancy_grid'))

    x = numpy.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x 


x = array(gauss(append(A,B,1)))
print('\nРешение методом Гаусса:\n')
print('\n'.join([''.join(["[{:.8f}]".format(item)])
      for item in x]))

print('\nРешение с помощью numpy.linalg.solve:\n')
print('\n'.join([''.join([format(item)]) 
      for item in numpy.linalg.solve(A,B)]))



print('\nОбратная матрица системы:\n')
print(tabulate(inv(A), tablefmt='fancy_grid'))

from numpy.linalg import norm

am = norm(inv(A))*0.001
rm = am/(norm(A)*norm(inv(A)))

print("\nАбсолютная погрешность равна {:>36}\nОтносительная погрешность больше, чем {:>27}\n".format(am,rm))