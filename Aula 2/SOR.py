import numpy as np
import matplotlib.pyplot as plt

#SOR

import numpy as np
import matplotlib.pyplot as plt


A = np.array([[ 2, -1,  0],
              [-1,  2, -1],
              [ 0, -1,  2]], dtype=float)

b = np.array([1, 0, 1], dtype=float)


def sor(A, b, omega=1.0, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    erros = []

    for k in range(max_iter):
        x_antigo = x.copy()
        for i in range(n):
            soma = b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]
            x_gs = soma / A[i, i]
            x[i] = (1 - omega) * x_antigo[i] + omega * x_gs
        erro = np.linalg.norm(x - x_antigo)
        #erro = np.max(np.abs(x - x_antigo))
        erros.append(erro)
        if erro < tol:
            return x, k + 1, True, erros

    return x, max_iter, False, erros


omega = 1.1
sol, iters, convergiu, erros = sor(A, b, omega=omega)

print(f"omega      = {omega}")
print(f"Solução:   {sol}")
print(f"Iterações: {iters}")
print(f"Convergencia : {convergiu}")


# Gráfico
plt.figure
plt.semilogy(range(1, len(erros) + 1), erros, 'o-',
             color='darkred')
plt.xlabel('Iteração $k$')
plt.ylabel(r'$\|x^{(k)} - x^{(k-1)}\|_\infty$')
plt.title(f'Convergência do SOR com $\\omega = {omega}$')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()











