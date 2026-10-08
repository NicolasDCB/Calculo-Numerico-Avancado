#MIX
import numpy as np
import matplotlib.pyplot as plt

#Exemplo 5.12

n = 10
A = np.zeros((n, n))

# diagonal
np.fill_diagonal(A, 3)
np.fill_diagonal(A[:-1, 1:], -1)
np.fill_diagonal(A[1:, :-1], -2)

# Vetor solução  iguais a 1
x_exato = np.ones(n)

b = A @ x_exato

def jacobi(A, b, tol=1e-12, max_iter=500):
    n = len(b)
    D = np.diag(np.diag(A))
    R = A - D

    x = np.zeros(n)
    historico = [x.copy()]

    for k in range(max_iter):
        x_novo = (b - R @ x) / np.diag(A)
        historico.append(x_novo.copy())

        erro = np.max(np.abs(x_novo - x))
        if erro < tol:
            return x_novo, k + 1, True, np.array(historico)
        x = x_novo

    return x, max_iter, False, np.array(historico)


def gauss_seidel(A, b, tol=1e-12, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    historico = [x.copy()]

    for k in range(max_iter):
        x_antigo = x.copy()

        for i in range(n):
            soma = b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]
            x[i] = soma / A[i, i]

        historico.append(x.copy())

        erro = np.max(np.abs(x - x_antigo))
        if erro < tol:
            return x, k + 1, True, np.array(historico)

    return x, max_iter, False, np.array(historico)

def sor(A, b, omega=0.00005, tol=1e-12, max_iter=500):
    n = len(b)

    if not (0 < omega < 2):
        raise ValueError("omega deve estar em (0, 2).")

    x = np.zeros(n)
    historico = [x.copy()]

    for k in range(max_iter):
        x_antigo = x.copy()

        for i in range(n):
            soma = b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]
            x_gs = soma / A[i, i]
            x[i] = (1 - omega) * x_antigo[i] + omega * x_gs

        historico.append(x.copy())

        erro = np.max(np.abs(x - x_antigo))
        if erro < tol:
            return x, k + 1, True, np.array(historico)

    return x, max_iter, False, np.array(historico)

# Executando os três métodos
tol = 1e-12

sol_j, it_j, conv_j, hist_j = jacobi(A, b, tol=tol)
sol_gs, it_gs, conv_gs, hist_gs = gauss_seidel(A, b, tol=tol)
sol_sor, it_sor, conv_sor, hist_sor = sor(A, b, omega=1.2, tol=tol)

# Solução exata para comparação
x_exato = np.linalg.solve(A, b)


print("=" * 55)
print(f"Solução exata: {x_exato}")
print("=" * 55)

print(f"\nJacobi:")
print(f"  Solução:   {sol_j}")
print(f"  Iterações: {it_j}")
print(f"  Convergencia  {conv_j}")

print(f"\nGauss-Seidel:")
print(f"  Solução:   {sol_gs}")
print(f"  Iterações: {it_gs}")
print(f"  Convergencia  {conv_gs}")

print(f"\nSOR (omega = ?):")
print(f"  Solução:   {sol_sor}")
print(f"  Iterações: {it_sor}")
print(f"  Convergencia  {conv_sor}")

# Erro verdadeiro ao longo das iterações

erro_j  = np.array([np.max(np.abs(h - x_exato)) for h in hist_j])
erro_gs = np.array([np.max(np.abs(h - x_exato)) for h in hist_gs])
erro_sor = np.array([np.max(np.abs(h - x_exato)) for h in hist_sor])

# Gráfico:

plt.figure (figsize=(9, 5))
plt.semilogy(erro_j,  label=f'Jacobi ({it_j} it.)')
plt.semilogy(erro_gs,  label=f'Gauss-Seidel ({it_gs} it.)')
plt.semilogy(erro_sor, label=f'SOR ω=1.1 ({it_sor} it.)')

plt.xlabel('Iteração')
plt.ylabel('Erro ')
plt.title(' Jacobi , Gauss-Seidel , SOR')
plt.legend()
plt.tight_layout()
plt.show()


