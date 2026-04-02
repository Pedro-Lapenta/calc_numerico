import numpy as np

def sor(A, b, omega=1.05, tol=1e-5, kmax=30):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)
    x = np.zeros(n)

    if any(np.diag(A) == 0):
        return None, 0

    for k in range(kmax):
        x_new = np.copy(x)

        for i in range(n):
            soma1 = sum(A[i][j] * x_new[j] for j in range(i))
            soma2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_gs = (b[i] - soma1 - soma2) / A[i][i]
            x_new[i] = (1 - omega) * x[i] + omega * x_gs

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

print("SOR com omega = 1.0:")
x, iteracoes = sor(A, b, omega=1.0)
print(f"Resultado: {np.round(x, 4)}")
print(f"Número de iterações: {iteracoes}")

print("\nSOR com omega = 1.5:")
x, iteracoes = sor(A, b, omega=1.5)
print(f"Resultado: {np.round(x, 4)}")
print(f"Número de iterações: {iteracoes}")