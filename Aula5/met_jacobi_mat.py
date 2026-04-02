import numpy as np

def jacobi_matricial(A, b, tol=1e-5, kmax=30):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n) 

    D = np.diag(A)
    L_U = A - np.diag(D)

    if any(D == 0):
        return None, 0

    for k in range(kmax):
        x_new = (b - np.dot(L_U, x)) / D
        erro = np.linalg.norm(x_new - x) / np.linalg.norm(x_new)

        if erro < tol:
            return x_new, k + 1

        x = x_new

    return x, kmax

A = np.array([
    [15,  5, -5],
    [ 4, 10,  1],
    [ 2, -2,  8]
], dtype=float)
b = np.array([30, 23, -10], dtype=float)

x, iteracoes = jacobi_matricial(A, b)
print(f"Resultado: {np.round(x, 4)}")
print(f"Número de iterações: {iteracoes}")