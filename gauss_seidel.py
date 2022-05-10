from copy import copy
import numpy as np
import numpy.linalg as la
import pandas as pd


def gauss_seidel(A, b, x0, MAX_ITER=100, tol=1e-4):
    recolect_data = [x0]
    recolect_error = [1.0]

    n = len(A[0])
    x = np.zeros(n)

    k = 1
    while k <= MAX_ITER:
        # método de Gauss-Seidel
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x0[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        error_abs = la.norm(x - x0, np.inf)

        # recolectando los datos para la impresion
        recolect_data.append(copy(x))
        recolect_error.append(error_abs)

        # el error relativo es el criterio de terminacion
        if error_abs < tol*la.norm(x0, np.inf):
            break

        # actualiza los datos para la siguiente iteracion
        x0 = np.copy(x)
        k += 1
    else:
        print("Too many iterations")
    return np.c_[recolect_data, recolect_error]


def gauss_seidel_accel(A, b, x0, omega, MAX_ITER=100, tol=0.0003):
    recolect_data = [x0]
    recolect_error = [1.0]

    n = len(A[0])
    x = np.zeros(n)

    k = 1
    while k <= MAX_ITER:
        # método de Gauss-Seidel
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x0[i + 1:])

            z = (b[i] - s1 - s2) / A[i, i]
            # método de aceleracion
            x[i] = omega*z + (1 - omega)*x0[i]

        error_abs = la.norm(x - x0, np.inf)

        # recolectando los datos para la impresion
        recolect_data.append(copy(x))
        recolect_error.append(error_abs)

        # el error relativo es el criterio de terminacion
        if error_abs < tol*la.norm(x0, np.inf):
            break

        # actualiza los datos para la siguiente iteracion
        x0 = np.copy(x)
        k += 1
    else:
        print("Too many iterations")
    return np.c_[recolect_data, recolect_error]


 
A = np.array([[4, -1, -6, 0],
              [0, 9, 4, -2],
              [-5, -4, 10, 8],
              [1, 0, -7, 5]])

b = np.array([2, -12, 21, -6])

x0 = np.zeros(4)
x_exact = np.array([3, -2, 2, 1.0])

print("Gauss Seidel method")
recolected_data = gauss_seidel_accel(A, b, x0, 0.871052, MAX_ITER=2000)
ratio = np.zeros(len(recolected_data))
ratio[1:] = recolected_data[1:, 4]/recolected_data[:-1, 4]

recolected_data = np.c_[recolected_data, ratio]
show_data = pd.DataFrame(data=recolected_data, columns=["x_1", "x_2", "x_3", "x_4", "||e||_inf", "Ratio"])
pd.set_option("display.precision", 6)
print(show_data)