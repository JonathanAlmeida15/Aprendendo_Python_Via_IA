from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# Carregar dataset
dados = fetch_california_housing(as_frame=True)
tabela = dados.frame

print(tabela.head())
print(tabela.describe())

X = tabela.drop(columns=['MedHouseVal'])  # Entradas
y = tabela['MedHouseVal']                 # Saída

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

print("MSE (Erro Quadrático Médio):", mean_squared_error(y_test, y_pred))
print("R² (Coeficiente de Determinação):", r2_score(y_test, y_pred))