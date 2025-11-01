import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords

# ======== 1. Criar dataset simples =========
# Você pode depois substituir por um dataset real de reviews (IMDb, Amazon, etc.)
data = {
    'texto': [
        "O produto é ótimo, superou minhas expectativas!",
        "Horrível, chegou quebrado e não funciona.",
        "Amei o atendimento, tudo perfeito.",
        "Não recomendo, qualidade péssima.",
        "Excelente compra, voltarei a comprar.",
        "Muito ruim, perdi meu dinheiro.",
        "Entrega rápida e produto maravilhoso!",
        "Demorou demais e veio errado.",
        "Gostei bastante, recomendo!",
        "Terrível, nunca mais compro aqui.",
        "Muito bom, eu gostei"
    ],
    'sentimento': [
        "positivo", "negativo", "positivo", "negativo", "positivo",
        "negativo", "positivo", "negativo", "positivo", "negativo",
        "positivo"
    ]
}

df = pd.DataFrame(data)

# ======== 2. Pré-processamento =========
X = df['texto']
y = df['sentimento']

# Divide em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======== 3. Vetorização (Transformar texto em números) =========
nltk.download('stopwords')
stopwords_pt = stopwords.words('portuguese')
vectorizer = CountVectorizer(stop_words=stopwords_pt)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ======== 4. Modelo de IA =========
modelo = MultinomialNB()
modelo.fit(X_train_vec, y_train)

# ======== 5. Avaliação =========
y_pred = modelo.predict(X_test_vec)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# ======== 6. Matriz de confusão =========
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negativo', 'positivo'], yticklabels=['negativo', 'positivo'])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Análise de Sentimentos")
plt.show()

# ======== 7. Testar o modelo =========
while True:
    texto = input("\nDigite uma frase para analisar (ou 'sair'): ")
    if texto.lower() == 'sair':
        break
    texto_vec = vectorizer.transform([texto])
    sentimento_pred = modelo.predict(texto_vec)[0]
    print(f"💬 Sentimento detectado: {sentimento_pred.upper()}")
