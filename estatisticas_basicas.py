import numpy as np

# Criando um array de números aleatórios entre 1 e 100
dados = np.random.randint(1, 101, size=20)
print("Dados:", dados)

# Estatísticas básicas
media = np.mean(dados)
mediana = np.median(dados)
desvio = np.std(dados)
maximo = np.max(dados)
minimo = np.min(dados)

print(f"Média: {media}")
print(f"Mediana: {mediana}")
print(f"Desvio padrão: {desvio}")
print(f"Máximo: {maximo}")
print(f"Mínimo: {minimo}")
