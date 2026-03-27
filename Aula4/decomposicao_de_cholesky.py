import numpy as np

def valida_cholesky(A):
    # 1. É quadrada?
    if A.shape[0] != A.shape[1]:
        print("Falha: A matriz nao é quadrada.")
        return False
        
    # 2. É simetrica? (A == A transposta)
    if not np.allclose(A, A.T):
        print("Falha: A matriz nao é simetrica.")
        return False
        
    # 3. É positiva definida? (Todos os autovalores > 0)
    autovalores = np.linalg.eigvals(A)
    if np.any(autovalores <= 0):
        print("Falha: A matriz nao é definida positiva (possui autovalores <= 0).")
        return False
    
    print("sistema satisfaz as condicoes para utilizacao do metodo")

    return True

A = np.array([
    [16,  4,  8,  4],
    [ 4, 10,  8,  4],
    [ 8,  8, 12, 10],
    [ 4,  4, 10, 12]
], dtype=float)

b = np.array([32, 26, 38, 30], dtype=float)

def sol(A, b):

    if not valida_cholesky(A):
        return None

    n = len(A)
    G = np.zeros((n,n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                soma = sum(G[i][k]**2 for k in range(j))
                G[i][j] = np.sqrt(A[i][i] - soma)
            else:
                soma = sum(G[i][k] * G[j][k] for k in range(j))
                G[i][j] = (A[i][j] - soma)/G[j][j]

    print("--- Matriz G (Cholesky) ---")
    print(np.round(G, 4))
    print("---------------------------\n")

    y = np.zeros(n)
    for i in range(n):
        soma = sum(G[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - soma)/G[i][i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = sum(G[j][i] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - soma) / G[i][i]

    return x

resultado = sol(A, b)

if resultado is not None:
    print("Solucao: ", resultado)