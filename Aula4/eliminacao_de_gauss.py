import numpy as np

def valida_gauss(A):
    # É quadrada?
    if A.shape[0] != A.shape[1]:
        print("Falha: A matriz não é quadrada.")
        return False
        
    # É não singular (det != 0)?
    if np.isclose(np.linalg.det(A), 0):
        print("Falha: A matriz é singular (determinante nulo). O sistema não possui solução única.")
        return False
    
    print("sistema satisfaz as condições para utilização do método")                    
    return True

def sol(A, b):

    if not valida_gauss(A):
        return None

    n = len(b)

    A = A.astype(float)
    b = b.astype(float)

    print("--- Matriz Inicial [A | b] ---")
    print(np.column_stack((A, b)))
    print("-" * 30)

    # transforma matriz em trangular superior 
    for k in range(n - 1): # percorre a diagonal
        for i in range(k + 1, n):
            if A[k][k] == 0:
                return -1 # evita /0
            
            m = A[i][k] / A[k][k]

            for j in range(k, n):
                A[i][j] -= m * A[k][j]
                        
            b[i] -= m * b[k]

    # print matriz final - antes da substituição  
    print(np.round(np.column_stack((A, b)), 4))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]
        
    return x
        

A = np.array([
    [16,  4,  8,  4],
    [ 4, 10,  8,  4],
    [ 8,  8, 12, 10],
    [ 4,  4, 10, 12]
], dtype=float)

b = np.array([32, 26, 38, 30], dtype=float)

resultado = sol(A, b)

if resultado is not None:
    print(f"\nSolução Final (x): {np.round(resultado, 4)}")