import numpy as np

def valida_lu(A):
    # É quadrada?
    if A.shape[0] != A.shape[1]:
        print("Falha: A matriz não é quadrada.")
        return False
    
    # Menores principais lideres != 0
    n = len(A)
    for k in range(1, n + 1):
        sub_matrix = A[:k, :k]
        if np.isclose(np.linalg.det(sub_matrix), 0):
            print(f"Falha: O menor principal lider de ordem {k} tem determinante nulo.")
            return False

    print("sistema satisfaz as condicoes para utilizacao do metodo")            
    return True

def sol(A, b):

    if not valida_lu(A):
        return None, None, None

    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        
        for j in range(i, n):
            soma_u = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - soma_u

        for j in range(i + 1, n):
            if U[i][i] == 0:
                return -1
            
            soma_l = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - soma_l) / U[i][i]


    y = np.zeros(n)
    for i in range(n):
        soma_y = sum(L[i][j] * y[j] for j in range(i))
        y[i] = (b[i] - soma_y) / L[i][i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma_x = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - soma_x) / U[i][i]

    return x, L, U

A = np.array([
    [16,  4,  8,  4],
    [ 4, 10,  8,  4],
    [ 8,  8, 12, 10],
    [ 4,  4, 10, 12]
], dtype=float)

b = np.array([32, 26, 38, 30], dtype=float)

x, L, U = sol(A, b)

if x is not None:
    print("\n--- Matriz L ---")
    print(np.round(L, 4))
    print("\n--- Matriz U ---")
    print(np.round(U, 4))
    print(f"\nSolucao Final (x): {np.round(x, 4)}")