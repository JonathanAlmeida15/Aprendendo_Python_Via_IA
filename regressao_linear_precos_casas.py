from sklearn.datasets import fetch_california_housing
import pandas as pd

# Carregar dataset
dados = fetch_california_housing(as_frame=True)
tabela = dados.frame

print(tabela.head())
print(tabela.describe())
