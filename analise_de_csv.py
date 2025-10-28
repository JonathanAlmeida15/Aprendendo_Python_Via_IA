import pandas as pd

# Carregar dataset
url = "https://calmcode.io/static/data/titanic.csv"
titanic = pd.read_csv(url)

# Ver as primeiras linhas
print(titanic.head())

# Informações gerais
print(titanic.info())

# Estatísticas descritivas
print(titanic.describe())

# Tratar dados faltantes
print("Nulos por coluna:")
print(titanic.isnull().sum())

# Preenchendo idade faltante com a média
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())
