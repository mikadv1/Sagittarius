import numpy as np

def lup_decomposition(A):
    """Выполняет LUP разложение матрицы A"""
    
    n = A.shape[0]
    A = A.astype(float).copy()
    P = np.eye(n)
    
    for k in range(n):
        # Находим строку с максимальным элементом в текущем столбце
        pivot_row = k + np.argmax(np.abs(A[k:, k]))
        
        # Если максимальный элемент равен 0, матрица вырождена
        if A[pivot_row, k] == 0:
            raise ValueError("Матрица вырождена")
        
        # Перестановка строк в A и P
        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            P[[k, pivot_row]] = P[[pivot_row, k]]
        
        # Метод Гаусса
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]
            for j in range(k + 1, n):
                A[i, j] = A[i, j] - A[i, k] * A[k, j]
    
    # Извлекаем L и U из модифицированной матрицы A
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    
    return L, U, P

def forward_substitution(L, b):
    """Решает систему Lx = b методом прямой подстановки"""
    
    n = L.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    
    return x

def backward_substitution(U, b):
    """Решает систему Ux = b методом обратной подстановки"""
    
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(i+1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x

def solve_lup(A, b):
    """Решает систему уравнений Ax = b с помощью LUP разложения"""
    
    L, U, P = lup_decomposition(A)
    
    b_permuted = P @ b
    
    y = forward_substitution(L, b_permuted)
    
    x = backward_substitution(U, y)
    
    return x


if __name__ == "__main__":
    pass
