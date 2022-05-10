import numpy as np
from scipy.integrate import quad as integral
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la
from copy import copy


def gauss_seidel_tridiagonal(c, d, e, b, x0, MAX_ITER=1000, tol=1e-4):
    n = len(d)
    x = np.zeros(n)
    k = 1
    while k <= MAX_ITER:
        # método de Gauss-Seidel tridiagonal
        x[0] = (b[0] - e[0]*x0[1])/d[0]
        for i in range(1, n - 1):
            x[i] = (b[i] - c[i-1]*x[i-1] - e[i-1]*x0[i+1])/d[i]
        x[-1] = (b[-1] - c[-1]*x[-2])/d[-1]

        # el error relativo es el criterio de terminacion
        if la.norm(x - x0, np.inf) < tol*la.norm(x0, np.inf):
            print(k)
            return x

        # actualiza los datos para la siguiente iteracion
        x0 = np.copy(x)
        k += 1
    else:
        print("Too many iterations")

# Step 2 phi_i(x)


def phi_i(x, x_, i):
    if 0 <= x <= x_[i - 1]:
        return 0
    if x_[i - 1] < x <= x_[i]:
        return (x - x_[i - 1]) / (x_[i] - x_[i - 1])
    elif x_[i] < x <= x_[i + 1]:
        return (x_[i + 1] - x) / (x_[i + 1] - x_[i])
    else:
        return 0


def piecewise_linear_rayleight_ritz(x_: np.ndarray, p, q, f):
    """
    -d/dx(p(x)dy/dx) + q(x)y = f(x), for 0 <= x <= 1, y(0) = 0 and y(1) = 0
    0 = x0, x1, x2, ..., xn+1 = 1, n + 2 points
    """
    n = len(x_) - 2
    Q = np.zeros((7, n + 2))

    # Step 1 For i=0,...,n set h[i]=x[i+1]-x[i]=x1-x0 for an equipartition
    h_ = x_[1:] - x_[0:-1]
    # Step 3 For each i = 0, 1, 2, ..., n - 1 compute
    for i in range(1, n + 2):
        def f1(x): return (x_[i + 1] - x) * (x - x_[i]) * q(x)
        def f2(x): return (x - x_[i - 1])**2 * q(x)
        def f3(x): return (x_[i + 1] - x)**2 * q(x)
        def f4(x): return p(x)
        def f5(x): return (x - x_[i - 1]) * f(x)
        def f6(x): return (x_[i + 1] - x) * f(x)

        if i < n:
            Q[1, i] = 1 / h_[i]**2 * integral(f1, x_[i], x_[i + 1])[0]
        if i <= n:
            Q[2, i] = 1 / h_[i - 1]**2 * integral(f2, x_[i - 1], x_[i])[0]
            Q[3, i] = 1 / h_[i]**2 * integral(f3, x_[i], x_[i + 1])[0]
            Q[5, i] = 1 / h_[i - 1] * integral(f5, x_[i - 1], x_[i])[0]
            Q[6, i] = 1 / h_[i] * integral(f6, x_[i], x_[i + 1])[0]

        Q[4, i] = 1 / h_[i - 1]**2 * integral(f4, x_[i - 1], x_[i])[0]
    # Step 4, 5 build tridiagonal linear system
    c_ = -Q[4, 2:n + 1] + Q[1, 1:n]
    d_ = Q[4, 1:n + 1] + Q[4, 2:n + 2] + Q[2, 1:n + 1] + Q[3, 1:n + 1]
    e_ = np.copy(c_)

    b_ = Q[5, 1:n + 1] + Q[6, 1:n + 1]
    # Step 6, 7, 8, 9, 10 solve a symmetric tridiagonal linear system

    x0_ = np.zeros_like(b_)

    coeff = gauss_seidel_tridiagonal(c_, d_, e_, b_, x0_)
    # Step 11 Stop
    print("The procedure is complete")
    return coeff


def phi(x, x_, c):
    return np.dot(c, np.array([phi_i(x, x_, i) for i in range(1, len(x_) - 1)]))


# -d/dx(p(x)dy/dx) + q(x)y = f(x), for 0 <= x <= 1, y(0) = 0 and y(1) = 0
def p(x): return 1
def q(x): return np.pi**2
def f(x): return 2*np.pi**2*np.sin(np.pi*x)
def y(x): return np.sin(x*np.pi)


x_ = np.linspace(0, 1, 11)  # equipartition
c_ = piecewise_linear_rayleight_ritz(x_, p, q, f)

phi_ = np.array([phi(x_i, x_, c_) for x_i in x_])
y_ = np.array([y(xi) for xi in x_])
error_ = np.abs(phi_ - y_)
table_ = pd.DataFrame(np.column_stack((x_, phi_, y_, error_)), columns=["xi", "φ(xi)", "y(xi)", "|φ(xi) - y(xi)|"])

plt.plot(table_["xi"], table_["φ(xi)"], label="φ(xi)")
plt.plot(table_["xi"], table_["y(xi)"], label="y(xi)")
plt.legend()
plt.title("Rayleigh-Ritz method")
plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
pd.set_option("display.precision", 10)
print(table_)
