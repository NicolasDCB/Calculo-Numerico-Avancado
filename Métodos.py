import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Problema
# ============================================================

n = 50

# Matriz tridiagonal:
# -1 na diagonal inferior
#  2 na diagonal principal
# -1 na diagonal superior

A = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)

# Vetor b
b = np.ones(n)

# Solução exata
x_exato = np.linalg.solve(A, b)


# ============================================================
# Richardson estacionário
# ============================================================

def richardson(A, b, alpha=None, tol=1e-6, max_iter=10000):

    n = len(b)

    if alpha is None:
        autovalores = np.linalg.eigvalsh(A)

        lam_min = autovalores.min()
        lam_max = autovalores.max()

        alpha = 2.0 / (lam_min + lam_max)

        print(f"[Richardson] lambda_min = {lam_min:.6f}")
        print(f"[Richardson] lambda_max = {lam_max:.6f}")
        print(f"[Richardson] alpha_opt = {alpha:.6f}")

    x = np.zeros(n)

    historico = [x.copy()]

    for k in range(max_iter):

        r = b - A @ x

        x_novo = x + alpha * r

        historico.append(x_novo.copy())

        erro = np.max(np.abs(x_novo - x))

        if erro < tol:
            return x_novo, k + 1, True, np.array(historico)

        x = x_novo

    return x, max_iter, False, np.array(historico)


# ============================================================
# Richardson dinâmico / Gradiente
# ============================================================

def richardson_dinamico(A, b, tol=1e-6, max_iter=10000):

    n = len(b)

    x = np.zeros(n)

    historico = [x.copy()]

    for k in range(max_iter):

        r = b - A @ x

        rTr = r @ r
        rAr = r @ (A @ r)

        if rAr == 0:
            return x, k, True, np.array(historico)

        alpha_k = rTr / rAr

        x_novo = x + alpha_k * r

        historico.append(x_novo.copy())

        erro = np.linalg.norm(x_novo - x)

        if erro < tol:
            return x_novo, k + 1, True, np.array(historico)

        x = x_novo

    return x, max_iter, False, np.array(historico)


# ============================================================
# Gradiente Conjugado
# ============================================================

def gradiente_conjugado(A, b, x0=None, tol=1e-10, max_iter=None):

    n = len(b)

    if max_iter is None:
        max_iter = n

    x = np.zeros(n) if x0 is None else x0.copy()

    r = b - A @ x

    p = r.copy()

    historico = [x.copy()]

    residuos = [np.linalg.norm(r)]

    for k in range(max_iter):

        Ap = A @ p

        rTr = r @ r
        pAp = p @ Ap

        alpha = rTr / pAp

        x = x + alpha * p

        r = r - alpha * Ap

        historico.append(x.copy())

        residuos.append(np.linalg.norm(r))

        if np.linalg.norm(r) < tol:
            return x, k + 1, True, np.array(historico), residuos

        beta = (r @ r) / rTr

        p = r + beta * p

    return x, max_iter, False, np.array(historico), residuos


# ============================================================
# Executando os métodos
# ============================================================

sol_r, it_r, conv_r, hist_r = richardson(
    A, b, tol=1e-6
)

sol_g, it_g, conv_g, hist_g = richardson_dinamico(
    A, b, tol=1e-6
)

sol_cg, it_cg, conv_cg, hist_cg, res_cg = gradiente_conjugado(
    A, b, tol=1e-10
)


# ============================================================
# Resultados
# ============================================================

print("\n================ RESULTADOS ================\n")

print("n =", n)

print("\nPrimeiros valores da solução exata:")
print(x_exato[:10])

print("\nRichardson fixo:")
print("Iterações =", it_r)
print("Convergiu =", conv_r)
print("Erro =", np.linalg.norm(sol_r - x_exato))

print("\nRichardson dinâmico:")
print("Iterações =", it_g)
print("Convergiu =", conv_g)
print("Erro =", np.linalg.norm(sol_g - x_exato))

print("\nGradiente Conjugado:")
print("Iterações =", it_cg)
print("Convergiu =", conv_cg)
print("Erro =", np.linalg.norm(sol_cg - x_exato))


# ============================================================
# Resíduos
# ============================================================

res_r = [
    np.linalg.norm(b - A @ x)
    for x in hist_r
]

res_g = [
    np.linalg.norm(b - A @ x)
    for x in hist_g
]

res_cg_plot = [
    np.linalg.norm(b - A @ x)
    for x in hist_cg
]


# ============================================================
# Gráfico
# ============================================================

plt.figure(figsize=(8, 5))

plt.semilogy(
    res_r,
    'o-',
    label='Richardson fixo',
    markersize=3
)

plt.semilogy(
    res_g,
    's-',
    label='Richardson dinâmico',
    markersize=3
)

plt.semilogy(
    res_cg_plot,
    '^-',
    label='Gradiente Conjugado',
    markersize=3
)

plt.xlabel('Iteração $k$')
plt.ylabel(r'$\|r^{(k)}\|_2$')
plt.title('Comparação dos métodos - Matriz tridiagonal n=50')

plt.legend()
plt.tight_layout()
plt.show()