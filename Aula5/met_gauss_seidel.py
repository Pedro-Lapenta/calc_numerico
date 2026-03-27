import numpy as np

def gauss_seidel(A, b, tol=1e-2, kmax=10):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n) 

    if any(np.diag(A) == 0):
        return None

    for k in range(kmax):
        x_new = np.copy(x)

        for i in range(n):
            soma1 = sum(A[i][j] * x_new[j] for j in range(i))
            soma2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - soma1 - soma2) / A[i][i]

        erro = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)

        if erro < tol:
            return x_new

        x = x_new

    return x

A = np.array([
    [5, 1, 1],
    [3, 4, 1],
    [3, 3, 6]
], dtype=float)
b = np.array([5, 6, 0], dtype=float)

x = gauss_seidel(A, b)
print(f"Resultado: {np.round(x, 4)}")