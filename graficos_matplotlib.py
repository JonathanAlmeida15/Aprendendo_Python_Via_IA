import matplotlib.pyplot as plt
import seaborn as sns

# Histograma da idade
plt.hist(titanic['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribuição das idades")
plt.xlabel("Idade")
plt.ylabel("Contagem")
plt.show()

# Gráfico de sobreviventes por classe
sns.countplot(x='Pclass', hue='Survived', data=titanic)
plt.title("Sobreviventes por Classe")
plt.show()
