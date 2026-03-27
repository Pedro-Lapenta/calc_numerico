import numpy as np

def jacobi_richardson(A, b, tol=1e-2, kmax=10):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n)
    
    if any(np.diag(A) == 0):
        return None

    for k in range(kmax):
        x_new = np.zeros(n)

        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - soma) / A[i][i]

        erro = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)

        if erro < tol:
            return x_new

        x = x_new

    return x

A = np.array([
    [10, 2,  1],
    [ 1, 5,  1],
    [ 2, 3, 10]
], dtype=float)
b = np.array([7, -8, 6], dtype=float)

x = jacobi_richardson(A, b)
print(f"Resultado: {np.round(x, 4)}")