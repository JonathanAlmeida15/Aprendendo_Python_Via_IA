# detector_spam_nb.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. Carregar dataset ===
# Dataset de exemplo (você pode substituir por outro CSV se quiser)
# Este é o dataset público de SMS Spam
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Amostra de dados:")
print(df.head())

# === 2. Converter rótulos ===
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# === 3. Dividir dados em treino e teste ===
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# === 4. Vetorização de texto ===
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 5. Treinar modelo ===
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# === 6. Fazer previsões ===
y_pred = model.predict(X_test_vec)

# === 7. Avaliação ===
print("\nAcurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))

# === 8. Testar com mensagens novas ===
testes = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",
    "Oi, tudo bem? Te vejo no almoço amanhã?",
    "Get cheap loans with no credit check! Apply now!",
    "Reunião confirmada às 15h com o cliente."
]

testes_vec = vectorizer.transform(testes)
previsoes = model.predict(testes_vec)

print("\n=== Testes manuais ===")
for msg, pred in zip(testes, previsoes):
    print(f"\nMensagem: {msg}\n→ Classificação: {'SPAM' if pred == 1 else 'NÃO SPAM'}")
