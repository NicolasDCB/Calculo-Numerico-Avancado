import numpy as np
import matplotlib.pyplot as plt

# Exemplo 1

A = np.array([[4, -1, 1],
              [1, 4, -1],
              [-1, 1, 4]], dtype=float)

b = np.array([5, 6, 13], dtype=float)

# Exemplo 2


A = np.array([[2, -1, 0],
              [-1, 2, -1],
              [0, -1, 2]], dtype=float)


b = np.array([1, 0, 1], dtype=float)

# GAUSS SEIDEL

A = np.array([[4, -1, 1],
              [1, 4, -1],
              [-1, 1, 4]], dtype=float)

b = np.array([5, 6, 13], dtype=float)

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    erros = []

    for k in range(max_iter):
        x_antigo = x.copy()
        for i in range(n):
            soma = b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]
            x[i] = soma / A[i, i]
        #rro = np.max(np.abs(x - x_antigo))
        erro = np.linalg.norm(x - x_antigo)

        erros.append(erro)
        if erro < tol:
            return x, k + 1, True, erros

    return x, max_iter, False, erros

sol, iters, convergiu, erros = gauss_seidel(A, b)
print(f"Solução:   {sol}")
print(f"Iterações: {iters}")
print(f"Convergencia : {convergiu}")


plt.figure
plt.semilogy(range(1, len(erros) + 1), erros, 'o-',
             color='navy'  )
plt.xlabel('Iteração $k$' )
plt.ylabel(r'$\|x^{(k)} - x^{(k-1)}\|_\infty$' )
plt.title('Convergência do Método de Gauss–Seidel' )
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()




