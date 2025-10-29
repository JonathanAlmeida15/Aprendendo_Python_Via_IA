import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Carregar o dataset
url = "https://calmcode.io/static/data/titanic.csv"
titanic = pd.read_csv(url)

# 2️⃣ Tratar valores ausentes
titanic['age'] = titanic['age'].fillna(titanic['age'].mean())

# 3️⃣ Criar gráficos
# Histograma da idade
plt.hist(titanic['age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribuição das idades")
plt.xlabel("Idade")
plt.ylabel("Contagem")
plt.show()

# Gráfico de sobreviventes por classe
sns.countplot(x='pclass', hue='survived', data=titanic)
plt.title("Sobreviventes por Classe")
plt.show()
print(titanic.columns)
