import numpy as np
import numpy.linalg as la


def gauss_seidel(A, b, x0, MAX_ITER=100, tol=1e-5):
    n = len(A[0])
    x = np.zeros(n)
    print(f"x[0] = {x0}")

    for k in range(1, MAX_ITER):
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x0[i + 1:])
            x[i] = (b[i] - s1 - s2) / A[i, i]

        print(f"x[{k}] = {x}")

        if la.norm(x - x0, np.inf) < tol*la.norm(x0, np.inf):
            return x, k

        x0 = np.copy(x)
    print("Too many iterations")


A = np.array([[10, 3, 1],
              [2, -10, 3],
              [1, 3, 10.]])

b = np.array([14.0, -5, 14])

x0 = np.array([0.0, 0.0, 0.0])

print("Gauss Seidel method")
gauss_seidel(A, b, x0)
