import numpy as np

# Matriz de coeficientes (corrigida a sintaxe da lista)
A = np.array([[10, -1, 2], 
              [-1, 11, -1], 
              [2, -1, 10]], dtype=float)

# Vetor dos termos independentes (exemplo)
b = np.array([6, 25, -11], dtype=float)

def metodo_jacobi(A, b, tol=1e-5, max_iter=100):
    n = len(b)
    x = np.zeros(n)          # Chute inicial: x^(0) = [0, 0, 0]
    x_novo = np.zeros(n)     # Vetor para armazenar a próxima iteração
    
    print(f"Chute inicial x_0: {x}\n")
    
    for k in range(max_iter):
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if i != j:
                    soma += A[i, j] * x[j]
            
            # Fórmula de atualização do Método de Jacobi:
            # x_i^(k+1) = (b_i - soma_(j!=i) (a_ij * x_j^(k))) / a_ii
            x_novo[i] = (b[i] - soma) / A[i, i]
            
        # Calcula a diferença máxima para testar o critério de parada (tolerância)
        erro = np.max(np.abs(x_novo - x))
        print(f"Iteração {k+1}: x = {np.round(x_novo, 5)}, Erro = {erro:.6f}")
        
        if erro < tol:
            print(f"\nConvergência atingida na iteração {k+1}!")
            return x_novo
        
        # Atualiza o vetor de soluções para a próxima iteração
        x = np.copy(x_novo)
        
    print("\nNúmero máximo de iterações atingido sem convergência.")
    return x_novo

# Executa o método
solucao = metodo_jacobi(A, b)
print("\nSolução final encontrada:", np.round(solucao, 4))