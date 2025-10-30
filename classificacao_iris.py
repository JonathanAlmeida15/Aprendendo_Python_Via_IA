import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
iris = data.frame
iris['species'] = iris['target'].apply(lambda i: data.target_names[i])


print(iris.head())

sns.pairplot(iris, hue="species")
plt.show()

X = iris.drop("species", axis=1)
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("\n=== Resultados KNN ===")
print("Acurácia:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

svm = SVC(kernel="linear")
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

print("\n=== Resultados SVM ===")
print("Acurácia:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, cmap="Blues", fmt="d", ax=axes[0])
axes[0].set_title("KNN - Matriz de Confusão")

sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, cmap="Greens", fmt="d", ax=axes[1])
axes[1].set_title("SVM - Matriz de Confusão")

plt.show()
