import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Carregar dataset (Mall Customers)
url = "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/Mall_Customers.csv"
df = pd.read_csv(url)

print("✅ Dataset carregado com sucesso!")
print(df.head(), "\n")

# 2. Selecionar colunas numéricas para análise
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 3. Normalizar os dados (importante para K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Descobrir o número ideal de clusters (método do cotovelo)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.show()

# 5. Treinar modelo com número ótimo (ex: 5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 6. Visualizar os clusters
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster',
    data=df,
    palette='tab10'
)
plt.title('Agrupamento de Clientes - K-Means')
plt.show()

# 7. Exibir médias de cada cluster
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
print("\n📊 Médias por cluster:\n", cluster_summary)
